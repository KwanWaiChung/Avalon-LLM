import random
import json
import os

from vllm import LLM, SamplingParams
from strictfire import StrictFire
from vllm_gameplay import RequestOutput
from copy import deepcopy
from fastchat.conversation import get_conv_template
from src.server.tasks.avalon.agents.my_vllm_agent import VllmAgent
from src.server.tasks.avalon.my_prompts import (
    GUESS_ONE_ROLE_PROMPT,
    GUESS_ROLE_CHEAT_DIFFERENT_HINT,
    GUESS_ROLE_CHEAT_SAME_HINT,
)
from src.utils.vllm_misc import RequestStatus
from typing import Dict, Any
from src.utils.logger import get_logger


class Request:
    def __init__(
        self,
        prompt: str,
        resp: str,
        game_idx: int,
        player_idx: int,
        round_idx: int,
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
    max_trial: int = 10,
):
    logger = get_logger(
        __name__,
        logger_level="debug",
        console_level="debug",
    )

    reqs = []
    data = [json.loads(row) for row in open(in_fn)]
    agent = VllmAgent(
        chat_template=get_conv_template("llama-3"),
        add_strategy_in_prompt=False,
        use_summary=True,
        max_trials=100,
    )
    for game_i, history in enumerate(data):
        for round_i in range(len(history["leaders"])):
            for player_idx, role in enumerate(history["roles"]):
                role_name = role[1]
                if role_name == "Merlin":
                    continue
                elif role[2]:  # Good player
                    # randomly pick another player
                    # choose all roles
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
                    new_history = {
                        "leaders": history["leaders"][: round_i + 1],
                        "team_discs": history["team_discs"][: round_i + 1],
                        "team_props": history["team_props"][:round_i],
                        "team_votes": history["team_votes"][:round_i],
                        "quest_votes": history["quest_votes"][:round_i],
                        "role_guess": history["role_guess"][:round_i],
                        "role_belief": history["role_belief"][:round_i],
                        "summaries": history["summaries"][:round_i],
                        "assassin": history["assassin"],
                        "roles": history["roles"],
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "n_error": 0,
                        "id": history["id"],
                    }
                    req = Request(
                        prompt=None,
                        resp=None,
                        game_idx=game_i,
                        player_idx=player_idx,
                        round_idx=round_i,
                        history=new_history,
                        status=RequestStatus.ROLE_GUESS_GET_PROMPT,
                    )
                    prompts, status = agent.guess_role_dpo_wo_env(
                        req=req,
                        to_guess_multiple_player=False,
                    )
                    req.prompt = prompts[0]
                    req.status = status
                    req.buffer["is_dpo"] = False
                    reqs.append(req)
                    dpo_req = Request(
                        prompt=prompts[1],
                        resp=None,
                        game_idx=game_i,
                        player_idx=player_idx,
                        round_idx=round_i,
                        history=new_history,
                        status=status,
                        buffer={k: v for k, v in req.buffer.items()},
                    )
                    dpo_req.buffer["is_dpo"] = True
                    reqs.append(dpo_req)
    model = LLM(model=model_name, dtype="float16", tensor_parallel_size=n_gpus)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    resps = model.generate(
        [req.prompt for req in reqs],
        sampling_params=sampling_params,
    )
    #  # debug
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
    res = {}
    while reqs:
        new_reqs = []
        for req in reqs:
            prompt, status = agent.guess_role_dpo_wo_env(
                req=req,
                to_guess_multiple_player=False,
            )
            if status == RequestStatus.ROLE_GUESS_SUCCEED:
                if not req.buffer["is_dpo"]:
                    res.setdefault(req.game_idx, {}).setdefault(
                        req.player_idx, {}
                    ).setdefault(req.round_idx, {}).setdefault(
                        req.buffer["tgt_role"], {}
                    )[
                        "normal"
                    ] = {
                        "resp": req.resp,
                        "tgt_player_i": req.buffer["tgt_player_i"],
                        "tgt_real_role": req.history["roles"][
                            req.buffer["tgt_player_i"]
                        ][1],
                        "prompt": prompt,
                    }
                else:
                    res.setdefault(req.game_idx, {}).setdefault(
                        req.player_idx, {}
                    ).setdefault(req.round_idx, {}).setdefault(
                        req.buffer["tgt_role"], {}
                    )[
                        "dpo"
                    ] = {
                        "resp": req.resp,
                        "tgt_player_i": req.buffer["tgt_player_i"],
                        "tgt_real_role": req.history["roles"][
                            req.buffer["tgt_player_i"]
                        ][1],
                        "prompt": prompt,
                    }
            elif req.buffer["trial"] < max_trial:
                # create another req
                new_reqs.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        round_idx=req.round_idx,
                        history=req.history,
                        status=status,
                        buffer=req.buffer,
                    )
                )
            else:
                logger.info(
                    f"Request of game_idx={req.game_idx}, player_idx={req.player_idx}, round_idx={req.round_idx}, tgt_role={req.tgt_role} has exceeded the maximum number of trials {max_trial} and will be ignored."
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

    # done and save
    with open(out_fn, "w") as f:
        json.dump(res, f, indent=4)

    print(f"Saved file to {out_fn}.")


if __name__ == "__main__":
    StrictFire(main)
