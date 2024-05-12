# imports
import torch
import random
import json
from accelerate import Accelerator
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
    PPOTrainer,
)
from strictfire import StrictFire
from peft import PeftModel, LoraConfig, get_peft_model
from fastchat.conversation import get_conv_template


# get models
def get_model(model_name: str, lora_path: str = None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map="auto",
        # device_map={"": Accelerator().local_process_index},
        torch_dtype=torch.float16,
        # torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    else:
        peft_kwargs = {
            "r": 8,
            "target_modules": "q_proj,v_proj",
            "lora_alpha": 16,
        }
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            **peft_kwargs,
        )
        model = get_peft_model(model, lora_config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model_ref = create_reference_model(model)
    return model, model_ref


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_data(filename: str):
    data = [json.loads(row) for row in open(filename)]
    seeder = random.Random(111)
    out_data = []
    for row in data:
        for round in row["role_guess"]:
            for item in round:
                tgt_role: str = row["roles"][item["tgt_player"]][1]
                role = seeder.choice(list(item["output"].keys()))
                res = item["output"][role]
                if role == tgt_role or (
                    role == "Minion" and tgt_role == "Assassin"
                ):
                    # 5 for score=10, -5 for score=0
                    reward = 5 - (10 - res["score"])
                else:
                    # -5 for score=10, 5 for score=0
                    reward = -(5 - (10 - res["score"]))
                prompt = (
                    item["prompt"]
                    .replace("Merlin and Servant", role)
                    .replace("Merlin, Servant, and Minion", role)
                )
                prompt = prompt.split("{")[0]
                prompt += f"""{{
    "{role}": {{
        "rationale": <str: rationale to support being {role}>,
        "score": <int: 1 to 10>
    }}
}}"""
                output = {role: item["output"][role]}
                response = json.dumps(output, ensure_ascii=False, indent=4)
                out_data.append((prompt, response, reward))
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
    lora_path: str = None,
    mini_batch_size: int = 1,
    grad_accum: int = 32,
    n_epochs: int = 4,
    n_samples: int = -1,  # -1 means all.
    save_path: str = "saves/avalon_ppo",
):
    model, model_ref = get_model(model_name, lora_path=lora_path)
    tokenizer = get_tokenizer(model_name)

    dataset = get_data(filename)

    prompt_tensors = []
    response_tensors = []
    reward_tensors = []
    for prompt, response, reward in dataset:
        template = get_conv_template("llama-3")
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()
        prompt_tensor = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
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

    if n_samples < 0:
        global_bz = mini_batch_size * grad_accum
        n_samples = (len(prompt_tensors) // global_bz) * global_bz
    prompt_tensors = prompt_tensors[:n_samples]
    response_tensors = response_tensors[:n_samples]
    reward_tensors = reward_tensors[:n_samples]

    # create a ppo trainer
    trainable_params, all_param = count_parameters(model)
    print(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )
    # initialize trainer
    ppo_config = PPOConfig(
        batch_size=len(prompt_tensors),
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=grad_accum,
        ppo_epochs=n_epochs,
        log_with="wandb",
        task_name="RoleGuessPPO",
    )
    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        model_ref,
        tokenizer,
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
        prompt_tensors, response_tensors, reward_tensors
    )
    ppo_trainer.save_pretrained(save_path)
    print(f"Trained model saved to {save_path}.")


if __name__ == "__main__":
    StrictFire(main)
