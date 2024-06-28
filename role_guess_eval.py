import random
import json
import os

from strictfire import StrictFire
from vllm_gameplay import RequestOutput
from copy import deepcopy
from fastchat.conversation import get_conv_template
from src.server.tasks.avalon.agents.my_vllm_agent import VllmAgent
from src.server.tasks.avalon.my_prompts import GUESS_ONE_ROLE_PROMPT
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
        tgt_player_i: int,
        tgt_role: str,
        history: Dict,
        status: int,
        to_forward: bool = True,
        args: Dict[str, Any] = None,
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
        if args is None:
            args = {}
        self.buffer = buffer
        self.args = args


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
    seed_global: bool = False,
    use_summary: bool = True,
    include_prev_disc: bool = True,
    guess_belief: bool = False,
    n_games: int = -1,
    debug: bool = False,
):
    if not debug:
        from vllm import LLM, SamplingParams
    seeder = random.Random(seed)
    logger = get_logger(
        __name__,
        logger_level="debug",
        console_level="debug",
        maxBytes=5e-6,
    )

    reqs = []
    data = [json.loads(row) for row in open(in_fn)]
    if n_games > 0:
        data = data[:n_games]
    for game_i, history in enumerate(data):
        for round_i in range(len(history["leaders"])):
            # tgt -> [src]
            tgt2src = {}
            for player_idx, role in enumerate(history["roles"]):
                role_name = role[1]
                tgt_player_i = seeder.choice(
                    [
                        i
                        for i in range(len(history["roles"]))
                        if i != player_idx
                    ]
                )
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
                else:
                    tgt_roles = ["Percival", "Servant", "Merlin"]
                tgt2src.setdefault(tgt_player_i, []).append(player_idx)
                for tgt_role in tgt_roles:
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
                        tgt_role=tgt_role,
                        tgt_player_i=tgt_player_i,
                        history=new_history,
                        status=RequestStatus.ROLE_GUESS_GET_PROMPT,
                    )
                    reqs.append(req)
            if guess_belief:
                merlin_idx = [
                    player_idx
                    for player_idx, role in enumerate(history["roles"])
                    if role[1] == "Merlin"
                ][0]
                all_roles = list(set([role[1] for role in history["roles"]]))
                for player_idx, src_role in enumerate(history["roles"]):
                    if player_idx not in tgt2src:
                        continue

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
                    # During role guess:
                    # 1. Evil player will only be guessed by Good Player excluding Merlin.
                    # 2. Good player will be guessed by anyone.
                    # there might no tgt_player available.

                    tgt_player_i: int = seeder.choice(tgt2src[player_idx])
                    for to_guess_role in all_roles:
                        req = Request(
                            prompt=None,
                            resp=None,
                            tgt_player_i=tgt_player_i,
                            tgt_role=to_guess_role,
                            game_idx=game_i,
                            player_idx=player_idx,
                            round_idx=round_i,
                            history=new_history,
                            status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
                            args={
                                "tgt_role": to_guess_role,
                                "tgt_player_i": tgt_player_i,
                            },
                        )
                        reqs.append(req)

                    # if not role[2]:
                    #     # if we want to pick tgt from good team
                    #     good_player_idxs = [
                    #         player_idx
                    #         for player_idx, role in enumerate(history["roles"])
                    #         if role[2]
                    #     ]
                    #     available_idxs = set(tgt2src[player_idx]) & set(
                    #         good_player_idxs
                    #     ) | set(
                    #         [
                    #             player_idx
                    #             for player_idx, role in enumerate(
                    #                 history["roles"]
                    #             )
                    #             if role[1] == "Merlin"
                    #         ]
                    #     )
                    #     tgt_player_i = seeder.choice(list(available_idxs))

    agent = VllmAgent(
        chat_template=get_conv_template("llama-3"),
        add_strategy_in_prompt=False,
        max_trials=100,
        seed=seed,
        use_summary=use_summary,
        include_prev_disc=include_prev_disc,
    )
    if not debug:
        model = LLM(
            model=model_name,
            dtype="float16",
            tensor_parallel_size=n_gpus,
            seed=seed if seed_global else 0,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed if not seed_global else None,
        )
    role_guess_res = {}
    role_belief_res = {}
    while reqs:
        new_reqs = []
        for req in reqs:
            if req.status in [
                RequestStatus.ROLE_GUESS_GET_PROMPT,
                RequestStatus.ROLE_GUESS_CHECK_ERROR,
            ]:
                prompt, status = agent.guess_role_for_eval(
                    req=req,
                    to_guess_multiple_player=False,
                )
                if status == RequestStatus.ROLE_GUESS_SUCCEED:
                    role_guess_res.setdefault(req.game_idx, {}).setdefault(
                        req.player_idx, {}
                    ).setdefault(req.round_idx, {})[req.tgt_role] = {
                        "score": int(req.resp["score"]),
                        "tgt_real_role": req.history["roles"][
                            req.tgt_player_i
                        ][1],
                        "tgt_player_i": req.tgt_player_i,
                        "src_role": req.history["roles"][req.player_idx][1],
                        "prompt": prompt,
                        "output": req.resp,
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
                            tgt_role=req.tgt_role,
                            tgt_player_i=req.tgt_player_i,
                            history=req.history,
                            status=status,
                            buffer=req.buffer,
                        )
                    )
                else:
                    logger.info(
                        f"Role guess request of game_idx={req.game_idx}, player_idx={req.player_idx}, round_idx={req.round_idx}, tgt_role={req.tgt_role} has exceeded the maximum number of trials {max_trial} and will be ignored."
                    )
            elif req.status in [
                RequestStatus.ROLE_BELIEF_GET_PROMPT,
                RequestStatus.ROLE_BELIEF_CHECK_ERROR,
            ]:
                # it's no env
                prompt, status = agent.guess_belief(
                    req=req,
                )
                to_guess_role = req.buffer["tgt_role"]
                tgt_player_i = req.buffer["tgt_player_i"]

                if status == RequestStatus.ROLE_BELIEF_SUCCEED:
                    tgt_player_i = req.buffer["tgt_player_i"]
                    role_belief_res.setdefault(req.game_idx, {}).setdefault(
                        req.player_idx, {}
                    ).setdefault(req.round_idx, {})[to_guess_role] = {
                        "score": int(req.resp["score"]),
                        "tgt_player_i": tgt_player_i,
                        "src_role": req.history["roles"][req.player_idx][1],
                        # "tgt_real_role": req.history["roles"][
                        #     req.tgt_player_i
                        # ][1],
                        "prompt": prompt,
                        "output": req.resp,
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
                            tgt_role=req.tgt_role,
                            tgt_player_i=req.tgt_player_i,
                            history=req.history,
                            status=status,
                            buffer=req.buffer,
                        )
                    )
                else:
                    logger.info(
                        f"Role belief request of game_idx={req.game_idx}, player_idx={req.player_idx}, round_idx={req.round_idx}, tgt_role={req.tgt_role} has exceeded the maximum number of trials {max_trial} and will be ignored."
                    )
        reqs = new_reqs

        # debug
        if debug:
            resps = [
                RequestOutput(
                    prompt=req.prompt,
                    outputs=[
                        json.dumps(
                            {
                                "score": seeder.randint(1, 10),
                                "rationale": "some rationale",
                            }
                        )
                    ],
                    prompt_len=1,
                    output_len=2,
                )
                for req in reqs
            ]
        else:
            # generate resps
            resps = model.generate(
                [req.prompt for req in reqs],
                sampling_params=sampling_params,
                # use_tqdm=False,
            )

        for req, resp in zip(reqs, resps):
            req.resp = resp.outputs[0].text

    # get the predicted role and calculate acc
    seeder = random.Random(seed)
    role_belief_acc = []
    # role guess
    n_samples = 0
    for game_idx, players in role_guess_res.items():
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
                role_belief_acc.append(
                    max_role == tgt_roles[max_role]["tgt_real_role"]
                )
                n_samples += 1
    print(f"Role guess has {n_samples} samples.")

    # belief guess
    if guess_belief:
        n_samples = 0
        role_belief_acc = []
        for game_idx, players in role_belief_res.items():
            for player_idx, rounds in players.items():
                for round_idx, tgt_roles in rounds.items():
                    max_score = max(
                        score["score"] for score in tgt_roles.values()
                    )
                    max_roles = [
                        key
                        for key, value in tgt_roles.items()
                        if value["score"] == max_score
                    ]
                    max_role = seeder.choice(max_roles)
                    tgt_player_i = tgt_roles[max_role]["tgt_player_i"]

                    gold_pred = role_guess_res[game_idx][tgt_player_i][
                        round_idx
                    ]
                    gold_role = gold_pred["pred_role"]
                    assert gold_pred[gold_role]["tgt_player_i"] == player_idx
                    role_belief_acc.append(max_role == gold_role)
                    n_samples += 1
        print(f"Role belief has {n_samples} samples.")

    # done and save
    with open(out_fn, "w") as f:
        json.dump(role_guess_res, f, indent=4)

    print(
        f"Saved file to {out_fn}. Acc= {sum(role_belief_acc)/len(role_belief_acc):.2f}"
    )

    # Merlin won't guess
    # good players pick any other
    # bad players pick good player


if __name__ == "__main__":
    StrictFire(main)
