import random
import json

from vllm import LLM, SamplingParams
from strictfire import StrictFire
from vllm_gameplay import RequestOutput
from copy import deepcopy
from fastchat.conversation import get_conv_template
from src.server.tasks.avalon.agents.my_vllm_agent import VllmAgent
from src.server.tasks.avalon.my_prompts import GUESS_ONE_ROLE_PROMPT
from src.utils.vllm_misc import RequestStatus
from typing import Dict, Any


class Request:
    def __init__(
        self,
        prompt: str,
        resp: str,
        game_idx: int,
        player_idx: int,
        round_idx: int,
        tgt_player_i: int,
        tgt_role: str,
        history: Dict,
        status: int,
        to_forward: bool = True,
        buffer: Dict[str, Any] = None,
    ):
        self.prompt = prompt
        self.resp = resp
        self.game_idx = game_idx
        self.player_idx = player_idx
        self.round_idx = round_idx
        self.tgt_player_i = tgt_player_i
        self.tgt_role = tgt_role
        self.history = history
        self.status = status
        self.to_forward = to_forward
        if buffer is None:
            buffer = {}
        self.buffer = buffer


def main(
    in_fn: str,
    out_fn: str,
    model_name: str,
    seed: int = 111,
    temperature=0,
    top_p=1,
    max_tokens: int = 512,
    n_gpus: int = 1,
):
    seeder = random.Random(seed)

    reqs = []
    data = [json.loads(row) for row in open(in_fn)]
    for game_i, history in enumerate(data):
        for round_i in range(len(history["leaders"])):
            for player_idx, role in enumerate(history["roles"]):
                role_name = role[1]
                if role_name == "Merlin":
                    continue
                elif role[2]:  # Good player
                    # randomly pick another player
                    # choose all roles
                    player_i = seeder.choice(
                        [
                            i
                            for i in range(len(history["roles"]))
                            if i != player_idx
                        ]
                    )
                    tgt_roles = [
                        "Percival",
                        "Servant",
                        "Servant",
                        "Morgana",
                        "Assassin",
                        "Merlin",
                    ]
                    tgt_roles.pop(tgt_roles.index(role_name))
                    tgt_roles = set(tgt_roles)
                    for tgt_role in tgt_roles:
                        new_history = {
                            "leaders": history["leaders"][: round_i + 1],
                            "team_discs": history["team_discs"][: round_i + 1],
                            "team_props": history["team_props"][: round_i + 1],
                            "team_votes": history["team_votes"][: round_i + 1],
                            "quest_votes": history["quest_votes"][
                                : round_i + 1
                            ],
                            "role_guess": history["role_guess"][: round_i + 1],
                            "role_belief": history["role_belief"][
                                : round_i + 1
                            ],
                            "summaries": history["summaries"][: round_i + 1],
                            "assassin": history["assassin"],
                            "roles": history["roles"],
                            "id": history["id"],
                        }
                        req = Request(
                            prompt=None,
                            resp=None,
                            game_idx=game_i,
                            player_idx=player_idx,
                            round_idx=round_i,
                            tgt_role=tgt_role,
                            tgt_player_i=player_i,
                            history=new_history,
                            status=RequestStatus.ROLE_GUESS_GET_PROMPT,
                        )
                        reqs.append(req)
    agent = VllmAgent(
        chat_template=get_conv_template("llama-3"),
        add_strategy_in_prompt=False,
        use_summary=True,
    )
    model = LLM(model=model_name, dtype="float16", tensor_parallel_size=n_gpus)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    res = {}
    while reqs:
        new_reqs = []
        for req in reqs:
            prompt, status = agent.guess_role_wo_env(
                req=req,
                to_guess_multiple_player=False,
            )
            if status == RequestStatus.ROLE_GUESS_SUCCEED:
                res.setdefault(req.game_idx, {}).setdefault(
                    req.player_idx, {}
                ).setdefault(req.round_idx, {})[req.tgt_role] = {
                    "score": int(req.resp["score"]),
                    "tgt_player_i": req.tgt_player_i,
                    "tgt_real_role": req.history["roles"][req.tgt_player_i][1],
                }
            else:
                # create another req
                new_reqs.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        round_idx=req.round_idx,
                        tgt_role=req.tgt_role,
                        tgt_player_i=req.tgt_player_i,
                        history=req.history,
                        status=status,
                        buffer=req.buffer,
                    )
                )
        reqs = new_reqs

        # generate resps
        resps = model.generate(
            [req.prompt for req in reqs],
            sampling_params=sampling_params,
            # use_tqdm=False,
        )
        # # debug
        # resps = [
        #     RequestOutput(
        #         prompt=req.prompt,
        #         outputs=[
        #             json.dumps(
        #                 {
        #                     "score": seeder.randint(1, 10),
        #                     "rationale": "some rationale",
        #                 }
        #             )
        #         ],
        #         prompt_len=1,
        #         output_len=2,
        #     )
        #     for req in reqs
        # ]

        for req, resp in zip(reqs, resps):
            req.resp = resp.outputs[0].text

    # get the predicted role and calculate acc
    seeder = random.Random(123)
    acc = []
    for game_idx, players in res.items():
        for player_idx, rounds in players.items():
            for round_idx, tgt_roles in rounds.items():
                max_score = max(score["score"] for score in tgt_roles.values())
                max_roles = [
                    key
                    for key, value in tgt_roles.items()
                    if value["score"] == max_score
                ]
                max_role = seeder.choice(max_roles)
                tgt_roles["pred_role"] = max_role
                acc.append(max_role == tgt_roles[max_role]["tgt_real_role"])

    # done and save
    with open(out_fn, "w") as f:
        json.dump(res, f, indent=4)

    print(f"Saved file to {out_fn}. Acc= {sum(acc)/len(acc):.2f}")

    # Merlin won't guess
    # good players pick any other
    # bad players pick good player


if __name__ == "__main__":
    StrictFire(main)
