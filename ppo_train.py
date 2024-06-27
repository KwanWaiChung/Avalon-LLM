# imports
import torch
import random
import json
from accelerate import Accelerator
from typing import Tuple, Dict, Any
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from ppo_trainer import PPOTrainer
from strictfire import StrictFire
from peft import PeftModel, LoraConfig, get_peft_model
from fastchat.conversation import get_conv_template

TRAJ_WIN_REWARD = 1
TRAJ_LOSE_REWARD = -1
ROLE_GUESS_SCORE = {
    1: -5,
    2: -4,
    3: -3,
    4: -2,
    5: -1,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    10: 5,
}

ROLE_BELIEF_SCORE = {
    0: 5,
    1: 4,
    2: 3,
    3: 2,
    4: 1,
    5: -1,
    6: -2,
    7: -3,
    8: -4,
    9: -5,
}
accelerator = Accelerator()

is_deepspeed_used = accelerator.distributed_type == "DEEPSPEED" and hasattr(
    accelerator.state, "deepspeed_plugin"
)


class DebugDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        context_length=1500,
        resp_length=500,
    ):
        self.prompt_tensors = [torch.tensor([10] * context_length)] * n_samples
        self.response_tensors = [torch.tensor([10] * resp_length)] * n_samples
        self.reward_tensors = [torch.tensor(1.0)] * n_samples

    def __getitem__(self, index) -> Any:
        return (
            self.prompt_tensors[index],
            self.response_tensors[index],
            self.reward_tensors[index],
        )

    def __len__(self) -> int:
        return len(self.prompt_tensors)


class PPODataset(Dataset):
    def __init__(
        self,
        filename,
        mini_batch_size: int,
        grad_accum: int,
        tokenizer,
        n_games: int = -1,
        include_guess_role: bool = False,
        include_guess_belief: bool = False,
    ):
        data = [json.loads(row) for row in open(filename)]
        data = [convert_int_keys(row) for row in data]
        if n_games > 0:
            data = data[:n_games]
        out_data = []
        for row in data:
            good_win = row["final_result"]
            player_rewards = {
                i: (
                    TRAJ_WIN_REWARD
                    if good_win == role[-1]
                    else TRAJ_LOSE_REWARD
                )
                for i, role in enumerate(row["roles"])
            }
            roles = {i: role for i, role in enumerate(row["roles"])}
            evil_roles = [role[1] for role in row["roles"] if not role[2]]
            # team disc
            for round_i, disc_round in enumerate(row["team_discs"]):
                for player_i, disc_dict in disc_round.items():
                    # ["strategy", "response", "prompt"]
                    out_data.append(
                        (
                            disc_dict["prompt"],
                            json.dumps(
                                {
                                    "strategy": disc_dict["strategy"],
                                    "response": disc_dict["response"],
                                },
                                indent=4,
                                ensure_ascii=False,
                            ),
                            player_rewards[int(player_i)],
                        )
                    )
            # team props
            for round_i, (leader, prop_dict) in enumerate(
                zip(row["leaders"], row["team_props"])
            ):
                # ["rationale", "team", "prompt"]
                out_data.append(
                    (
                        prop_dict["prompt"],
                        json.dumps(
                            {
                                "rationale": prop_dict["rationale"],
                                "team": prop_dict["team"],
                            },
                            indent=4,
                            ensure_ascii=False,
                        ),
                        player_rewards[int(leader)],
                    )
                )

            # team votes
            for round_i, vote_round in enumerate(row["team_votes"]):
                for player_i, vote_dict in vote_round["votes"].items():
                    # ["rationale", "team", "prompt"]
                    out_data.append(
                        (
                            vote_dict["prompt"],
                            json.dumps(
                                {
                                    "rationale": vote_dict["rationale"],
                                    "vote": (
                                        "approve"
                                        if vote_dict["vote"]
                                        else "reject"
                                    ),
                                },
                                indent=4,
                                ensure_ascii=False,
                            ),
                            player_rewards[int(player_i)],
                        )
                    )

            # quest votes
            for round_i, vote_round in enumerate(row["quest_votes"]):
                for player_i, vote_dict in vote_round["votes"].items():
                    # ["rationale", "team", "prompt"]
                    out_data.append(
                        (
                            vote_dict["prompt"],
                            json.dumps(
                                {
                                    "rationale": vote_dict["rationale"],
                                    "vote": (
                                        "pass" if vote_dict["vote"] else "fail"
                                    ),
                                },
                                indent=4,
                                ensure_ascii=False,
                            ),
                            player_rewards[int(player_i)],
                        )
                    )

            # role_guess
            if include_guess_role:
                for round_i, guess_round in enumerate(row["role_guess"]):
                    for src_player_i, guess_dicts in guess_round.items():
                        tgt_players = set()
                        for guess_dict in guess_dicts:
                            # [output, prompt, src_player:int, tgt_player:int]
                            if guess_dict["tgt_player"] not in tgt_players:
                                # duplicate guesses exist in previous code
                                sign: bool = (
                                    guess_dict["tgt_role"]
                                    == roles[guess_dict["tgt_player"]][1]
                                )
                                out_data.append(
                                    (
                                        guess_dict["prompt"],
                                        json.dumps(
                                            guess_dict["output"],
                                            indent=4,
                                            ensure_ascii=False,
                                        ),
                                        int(sign)
                                        * ROLE_GUESS_SCORE[
                                            guess_dict["output"]["score"]
                                        ],
                                    )
                                )
                                tgt_players.add(guess_dict["tgt_player"])

            # role_belief
            if include_guess_belief:
                for round_i, belief_round in enumerate(row["role_belief"]):
                    for src_player_i, belief_dict in belief_round.items():
                        # [output, prompt, src_player:int, tgt_player:int]
                        tgt_player: int = belief_dict["tgt_player"]
                        tgt_role: Tuple[int, str, bool] = roles[
                            int(tgt_player)
                        ]
                        src_role: Tuple[int, str, bool] = roles[
                            int(src_player_i)
                        ]
                        to_guess_role: str = belief_dict["tgt_role"]
                        if src_role[2] and tgt_role[1] == "Merlin":
                            # Merlin's belief on good team member
                            gold_score = 10 if to_guess_role == src_role else 1
                        elif not src_role[2] and (
                            not tgt_role[2] or tgt_role[1] == "Merlin"
                        ):
                            # Evil player/Merlin's belief on evil player
                            gold_score = (
                                10 if to_guess_role in evil_roles else 1
                            )
                        else:
                            # Rely on role guess
                            guesses = []
                            for role_guess in row["role_guess"][round_i][
                                tgt_player
                            ]:
                                guesses.append(role_guess["tgt_player"])
                                if role_guess["tgt_player"] == int(
                                    src_player_i
                                ):
                                    gold_score: int = role_guess["output"][
                                        "score"
                                    ]
                                    break
                            else:
                                raise ValueError(
                                    f"`gold_score` not provided in role_guess. Player {tgt_player} only guesses Player {guesses}'s role, but not {src_player_i}."
                                )

                        out_data.append(
                            (
                                belief_dict["prompt"],
                                json.dumps(
                                    belief_dict["output"],
                                    indent=4,
                                    ensure_ascii=False,
                                ),
                                ROLE_BELIEF_SCORE[
                                    abs(
                                        belief_dict["output"]["score"]
                                        - gold_score
                                    )
                                ],
                            )
                        )
            # summaries
            for round_i, sum_round in enumerate(row["summaries"]):
                for player_i, sum_dict in sum_round.items():
                    # ["prompt", "resp"]
                    out_data.append(
                        (
                            sum_dict["prompt"],
                            sum_dict["resp"],
                            player_rewards[int(player_i)],
                        )
                    )

            # assassin
            if row["assassin"]:
                out_data.append(
                    (
                        row["assassin"]["prompt"],
                        json.dumps(
                            {
                                "rationale": row["assassin"]["rationale"],
                                "merlin": row["assassin"]["merlin"],
                            },
                            indent=4,
                            ensure_ascii=False,
                        ),
                        TRAJ_LOSE_REWARD if good_win else TRAJ_WIN_REWARD,
                    )
                )
        prompt_tensors = []
        response_tensors = []
        reward_tensors = []
        for prompt, response, reward in out_data:
            template = get_conv_template("llama-3")
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt()
            prompt_tensor = tokenizer(prompt, return_tensors="pt")[
                "input_ids"
            ][0]
            template.update_last_message(response)
            response = template.get_prompt()
            response_tensor = tokenizer(response, return_tensors="pt")[
                "input_ids"
            ][0]
            response_tensor = response_tensor[prompt_tensor.shape[0] :]
            reward_tensor = torch.tensor(reward, dtype=torch.float64)

            prompt_tensors.append(prompt_tensor)
            response_tensors.append(response_tensor)
            reward_tensors.append(reward_tensor)

        global_bz = mini_batch_size * grad_accum
        n_samples = (len(prompt_tensors) // global_bz) * global_bz
        self.prompt_tensors = prompt_tensors[:n_samples]
        self.response_tensors = response_tensors[:n_samples]
        self.reward_tensors = reward_tensors[:n_samples]

    def __getitem__(self, index) -> Any:
        return (
            self.prompt_tensors[index],
            self.response_tensors[index],
            self.reward_tensors[index],
        )

    def __len__(self) -> int:
        return len(self.prompt_tensors)


def convert_int_keys(d):
    new_dict = {}
    for k, v in d.items():
        key = int(k) if k.isdigit() else k
        if isinstance(v, dict):
            new_dict[key] = convert_int_keys(v)
        elif isinstance(v, list):
            new_dict[key] = [
                convert_int_keys(i) if isinstance(i, dict) else i for i in v
            ]
        else:
            new_dict[key] = v
    return new_dict


# get models
def get_model(
    model_name: str, lora_path: str = None, lora_config: Dict[str, Any] = None
):
    if (lora_path is None) == (lora_config is None):
        raise ValueError(
            "You must only either specify `lora_path` of `lora_config`."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None if is_deepspeed_used else "auto",
        # device_map={"": Accelerator().local_process_index},
        torch_dtype=torch.float16,
        # torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    else:
        # peft_kwargs = {
        #     "r": 8,
        #     "target_modules": "q_proj,v_proj",
        #     "lora_alpha": 16,
        # }
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            **lora_config,
        )
        model = get_peft_model(model, lora_config)

    rl_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    # return rl_model, None
    rl_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        model
    ).eval()
    # model_ref = create_reference_model(model)
    return rl_model, rl_model_ref


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_data(
    filename: str, include_guess_role: str, include_guess_belief: str
):
    data = [json.loads(row) for row in open(filename)]
    data = [convert_int_keys(row) for row in data]
    out_data = []
    for row in data:
        good_win = row["final_result"]
        player_rewards = {
            i: TRAJ_WIN_REWARD if good_win == role[-1] else TRAJ_LOSE_REWARD
            for i, role in enumerate(row["roles"])
        }
        roles = {i: role[1] for i, role in enumerate(row["roles"])}
        # team disc
        for disc_round in row["team_discs"]:
            for player_i, disc_dict in disc_round.items():
                # ["strategy", "response", "prompt"]
                out_data.append(
                    (
                        disc_dict["prompt"],
                        json.dumps(
                            {
                                "strategy": disc_dict["strategy"],
                                "response": disc_dict["response"],
                            },
                            indent=4,
                            ensure_ascii=False,
                        ),
                        player_rewards[int(player_i)],
                    )
                )
        # team props
        for leader, prop_dict in zip(row["leaders"], row["team_props"]):
            # ["rationale", "team", "prompt"]
            out_data.append(
                (
                    prop_dict["prompt"],
                    json.dumps(
                        {
                            "rationale": prop_dict["rationale"],
                            "team": prop_dict["team"],
                        },
                        indent=4,
                        ensure_ascii=False,
                    ),
                    player_rewards[int(leader)],
                )
            )

        # team votes
        for vote_round in row["team_votes"]:
            for player_i, vote_dict in vote_round["votes"].items():
                # ["rationale", "team", "prompt"]
                out_data.append(
                    (
                        vote_dict["prompt"],
                        json.dumps(
                            {
                                "rationale": vote_dict["rationale"],
                                "vote": (
                                    "approve"
                                    if vote_dict["vote"]
                                    else "reject"
                                ),
                            },
                            indent=4,
                            ensure_ascii=False,
                        ),
                        player_rewards[int(player_i)],
                    )
                )

        # quest votes
        for vote_round in row["quest_votes"]:
            for player_i, vote_dict in vote_round["votes"].items():
                # ["rationale", "team", "prompt"]
                out_data.append(
                    (
                        vote_dict["prompt"],
                        json.dumps(
                            {
                                "rationale": vote_dict["rationale"],
                                "vote": (
                                    "pass" if vote_dict["vote"] else "fail"
                                ),
                            },
                            indent=4,
                            ensure_ascii=False,
                        ),
                        player_rewards[int(player_i)],
                    )
                )

        # role_guess
        if include_guess_role:
            for guess_round in row["role_guess"]:
                for src_player_i, guess_dicts in guess_round.items():
                    tgt_players = set()
                    for guess_dict in guess_dicts:
                        # [output, prompt, src_player:int, tgt_player:int]
                        if guess_dict["tgt_player"] not in tgt_players:
                            # duplicate guesses exist in previous code
                            sign: bool = (
                                guess_dict["tgt_role"]
                                == roles[guess_dict["tgt_player"]]
                            )
                            out_data.append(
                                (
                                    guess_dict["prompt"],
                                    json.dumps(
                                        guess_dict["output"],
                                        indent=4,
                                        ensure_ascii=False,
                                    ),
                                    int(sign)
                                    * ROLE_GUESS_SCORE[
                                        guess_dict["output"]["score"]
                                    ],
                                )
                            )
                            tgt_players.add(guess_dict["tgt_player"])

        # role_belief
        if include_guess_belief:
            for round_i, belief_round in enumerate(row["role_belief"]):
                for src_player_i, belief_dict in belief_round.items():
                    # [output, prompt, src_player:int, tgt_player:int]
                    tgt_player: int = belief_dict["tgt_player"]
                    tgt_role: str = roles[int(tgt_player)]
                    src_role: str = roles[int(src_player_i)]
                    to_guess_role: str = belief_dict["tgt_role"]
                    if src_role == "Servant" and tgt_role == "Merlin":
                        if to_guess_role == "Servant":
                            gold_score = 10
                        else:
                            gold_score = 1
                    elif src_role in ["Minion", "Assassin"] and tgt_role in [
                        "Minion",
                        "Assassin",
                        "Merlin",
                    ]:
                        if to_guess_role in ["Minion", "Assassin"]:
                            gold_score = 10
                        else:
                            gold_score = 1
                    else:
                        guesses = []
                        for role_guess in row["role_guess"][round_i][
                            tgt_player
                        ]:
                            guesses.append(role_guess["tgt_player"])
                            if role_guess["tgt_player"] == int(src_player_i):
                                gold_score: int = role_guess["output"]["score"]
                                break
                        else:
                            raise ValueError(
                                f"`gold_score` not provided in role_guess. Player {tgt_player} only guesses Player {guesses}'s role, but not {src_player_i}."
                            )

                    sign: bool = (
                        belief_dict["tgt_role"]
                        == roles[belief_dict["tgt_player"]]
                    )
                    out_data.append(
                        (
                            belief_dict["prompt"],
                            json.dumps(
                                belief_dict["output"],
                                indent=4,
                                ensure_ascii=False,
                            ),
                            ROLE_BELIEF_SCORE[
                                abs(
                                    belief_dict["output"]["score"] - gold_score
                                )
                            ],
                        )
                    )
        # summaries
        for sum_round in row["summaries"]:
            for player_i, sum_dict in sum_round.items():
                # ["prompt", "resp"]
                out_data.append(
                    (
                        sum_dict["prompt"],
                        sum_dict["resp"],
                        player_rewards[int(player_i)],
                    )
                )

        # assassin
        if row["assassin"]:
            out_data.append(
                (
                    row["assassin"]["prompt"],
                    json.dumps(
                        {
                            "rationale": row["assassin"]["rationale"],
                            "merlin": row["assassin"]["merlin"],
                        },
                        indent=4,
                        ensure_ascii=False,
                    ),
                    TRAJ_LOSE_REWARD if good_win else TRAJ_WIN_REWARD,
                )
            )
    #         # here is the old code to handle multiple role guess
    #         for round in row["role_guess"]:
    #             for item in round:
    #                 tgt_role: str = row["roles"][item["tgt_player"]][1]
    #                 role = seeder.choice(list(item["output"].keys()))
    #                 res = item["output"][role]
    #                 if role == tgt_role or (
    #                     role == "Minion" and tgt_role == "Assassin"
    #                 ):
    #                     # 5 for score=10, -5 for score=0
    #                     reward = 5 - (10 - res["score"])
    #                 else:
    #                     # -5 for score=10, 5 for score=0
    #                     reward = -(5 - (10 - res["score"]))
    #                 prompt = (
    #                     item["prompt"]
    #                     .replace("Merlin and Servant", role)
    #                     .replace("Merlin, Servant, and Minion", role)
    #                 )
    #                 prompt = prompt.split("{")[0]
    #                 prompt += f"""{{
    #     "{role}": {{
    #         "rationale": <str: rationale to support being {role}>,
    #         "score": <int: 1 to 10>
    #     }}
    # }}"""
    #                 output = {role: item["output"][role]}
    #                 response = json.dumps(output, ensure_ascii=False, indent=4)
    #                 out_data.append((prompt, response, reward))
    return out_data


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(
                param.quant_storage, "itemsize"
            ):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def main(
    model_name,
    filename,
    run_name,
    lora_path: str = None,
    lora_config: Dict[str, Any] = None,
    lr: float = 1e-5,
    mini_batch_size: int = 1,
    loglikelihood_batch_size: int = 1,  # usually n_device * mini_bz
    grad_accum: int = 32,
    n_epochs: int = 4,
    n_games: int = -1,  # -1 means all.
    include_guess_role: bool = False,
    include_guess_belief: bool = False,
    save_path: str = "saves/avalon_ppo",
    debug: bool = False,
):

    tokenizer = get_tokenizer(model_name)
    if debug:
        dataset = DebugDataset(n_samples=n_games * 4)
    else:
        dataset = PPODataset(
            filename,
            mini_batch_size=mini_batch_size,
            grad_accum=grad_accum,
            n_games=n_games,
            tokenizer=tokenizer,
            include_guess_role=include_guess_role,
            include_guess_belief=include_guess_belief,
        )
    print(f"Number of training samples: {len(dataset)}")
    # create a ppo trainer
    model, model_ref = get_model(
        model_name, lora_path=lora_path, lora_config=lora_config
    )
    trainable_params, all_param = count_parameters(model)
    print(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )
    # initialize trainer
    ppo_config = PPOConfig(
        batch_size=len(dataset),
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=grad_accum,
        ppo_epochs=n_epochs,
        tracker_project_name="avalon",
        tracker_kwargs={"wandb": {"name": run_name}},
        log_with="wandb",
        learning_rate=lr,
    )
    if accelerator.distributed_type == "DEEPSPEED":
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = mini_batch_size
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=model_ref,
        tokenizer=tokenizer,
        dataset=dataset,
        loglikelihood_batch_size=loglikelihood_batch_size,
    )

    # train model for one step with ppo
    # queries (List[`torch.LongTensor`]):
    #     List of tensors containing the encoded queries of shape (`query_length`)
    # responses (List[`torch.LongTensor`]):
    #     List of tensors containing the encoded responses of shape (`response_length`)
    # scores (List[`torch.FloatTensor`]):
    #     List of tensors containing the scores. These are scalar.
    # response_masks (List[`torch.FloatTensor`], *optional*)):
    #     List of tensors containing masks of the response tokens.
    train_stats = ppo_trainer.step(
        dataset.prompt_tensors,
        dataset.response_tensors,
        dataset.reward_tensors,
    )
    # ppo_trainer.log_stats(
    #     stats=train_stats,
    #     batch={},
    #     rewards=dataset.reward_tensors,
    #     columns_to_log=[],
    # )
    ppo_trainer.save_pretrained(save_path)
    if accelerator.is_main_process:
        print(f"Trained model saved to {save_path}.")


if __name__ == "__main__":
    StrictFire(main)
