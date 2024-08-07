import json
import random
import os
import pickle
import torch
import numpy as np
from collections import defaultdict
from src.server.tasks.avalon.engine import (
    AvalonGameEnvironment,
    AvalonBasicConfig,
)
from src.utils.vllm_misc import Request, RequestStatus
from src.server.tasks.avalon.agents.my_vllm_agent import VllmAgent
from src.utils.inference import (
    DummyInferenceStrategy,
    AnyscaleInferenceStrategy,
    TogetherInferenceStrategy,
    LocalInferenceStrategy,
    OpenAIInferenceStrategy,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.conversation import get_conv_template
from src.utils.logger import get_logger
from typing import List, Dict, Union, Tuple
from strictfire import StrictFire
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
from src.utils.constants import (
    ROLE_GUESS_RESULT_DIR,
    MODELS,
    ROLE_GUESS_OUTPUT_DIR,
)


USE_MESSAGE = True
DEBUG = 0
strategy = OpenAIInferenceStrategy(key_path=None)


def wrapper(
    prompt,
    model,
    max_tokens: int,
    temperature: float,
    top_p: float,
    base_url: str,
):
    if USE_MESSAGE:
        output = strategy.generate(
            model_name=model,
            messages=prompt,
            max_tokens=max_tokens,
            chat_mode=True,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            seed=None,
        )
    else:
        output = strategy.generate(
            model_name=model,
            prompt=prompt,
            max_tokens=max_tokens,
            chat_mode=False,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            seed=None,
        )
    return output


# for debugging purpose only
def _get_all_prev(req, verbose=True):
    res = []
    while req.prev:
        req = req.prev
        k = f"{req.status}, player_idx={req.player_idx}, n_leaders={len(req.history['leaders'])}"
        res.append(req)
        if verbose:
            print(k)
    return res


class Output:
    def __init__(self, text: str):
        self.text = text


class RequestOutput:
    def __init__(
        self, prompt: str, outputs: List[str], prompt_len, output_len
    ):
        self.prompt = prompt
        self.outputs = [Output(text=o) for o in outputs]
        self.prompt_len = prompt_len
        self.output_len = output_len


class SamplingParamsWrapper:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens


class VllmStrategyWrapper:
    # this class wraps local inference pretending to be vllm
    def __init__(self, strategy, model_name, end_tokens=[]):
        self.strategy = strategy
        self.model_name = model_name
        self.end_tokens = end_tokens

    def generate(
        self, prompts: List[str], sampling_params, *args, **kwargs
    ) -> List[RequestOutput]:
        max_tokens = sampling_params.max_tokens
        temperature = sampling_params.temperature
        top_p = sampling_params.top_p
        outputs = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            output = self.strategy.generate(
                model_name=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                end_tokens=self.end_tokens,
            )
            outputs.append(
                RequestOutput(
                    prompt=prompt,
                    outputs=[output["output"]],
                    prompt_len=output["prompt_len"],
                    output_len=output["output_len"],
                )
            )
        return outputs


def construct_two_players_game(history, model_names, revert: bool = False):
    # game 1
    history["n_error"] = {model_name: 0 for model_name in model_names}
    history["models"] = [
        (model_names[0] if role[-1] == revert else model_names[1])
        for role in history["roles"]
    ]
    return history


class RequestProcessor:
    def __init__(
        self,
        agents: List[Tuple[str, VllmAgent]],
        to_discuss: bool,  # game arguments
        add_strategy_in_prompt: bool,
        to_guess_role: bool,
        to_guess_multiple_player_role: bool,
        n_guess_role_repeat: int,
        to_guess_belief: bool,
        use_summary: Union[bool, List[bool]],
        logger=None,
        use_team_prompt: bool = False,
    ):
        self.agents = {name: agent for name, agent in agents}
        self.to_discuss = to_discuss
        self.add_strategy_in_prompt = add_strategy_in_prompt
        self.to_guess_role = to_guess_role
        self.to_guess_multiple_player_role = to_guess_multiple_player_role
        self.n_guess_role_repeat = n_guess_role_repeat
        self.to_guess_belief = to_guess_belief
        self.use_summary = (
            use_summary if isinstance(use_summary, list) else [use_summary]
        )
        self.logger = logger
        self.n_finished_games = 0
        self.use_team_prompt = use_team_prompt

        # use to block and finish all role guess and belief before
        # heading to summary
        self.guess_and_belief_counter = defaultdict(int)
        self.n_summaries = defaultdict(int)

    def _set_curr_queue(self, queue: List[Request]):
        # for debugging only
        self.cur_queue = queue

    def _get_num_players(self, req) -> int:
        return len(req.history["roles"])

    def finish_game(self, req):
        if not req.env.done:
            raise ValueError("Game not yet done")

        n_rounds = len(req.history["leaders"])
        n_player = self._get_num_players(req)
        # team discs check
        if self.to_discuss:
            assert len(req.history["team_discs"]) == n_rounds
            for team_disc in req.history["team_discs"]:
                assert (
                    len(team_disc) == n_player
                ), f"Should have {n_player} players discussed in game {req.game_idx}."
                for player in team_disc.values():
                    for k in ["strategy", "response", "prompt"]:
                        assert (
                            k in player
                        ), f"Should have {k} in team discussion response, but received {player} in game {req.game_idx}."

        # team props check
        assert len(req.history["team_props"]) == n_rounds
        for team_prop in req.history["team_props"]:
            for k in ["rationale", "team", "prompt"]:
                assert (
                    k in team_prop
                ), f"{k} should be in `team_prop` but received {team_prop}"

        # team_votes check
        for team_vote in req.history["team_votes"]:
            assert (
                len(team_vote["votes"]) == n_player
            ), f"Should have {n_player} votes but received only {len(team_vote['votes'])} votes in game {req.game_idx}."

        # quest vote check
        success_teams = sum([v["result"] for v in req.history["team_votes"]])
        assert (
            sum([v["result"] is not None for v in req.history["quest_votes"]])
            == success_teams
        ), f"Number of non-none quest votes should equal to number of sucessed team in game {req.game_idx}"

        # summaries check
        # if self.use_summary:
        #     assert (
        #         len(req.history["summaries"]) == n_rounds
        #     ), f"Should have same number of summaries as number of rounds in game {req.game_idx}."
        #     for summary in req.history["summaries"]:
        #         assert (
        #             len(summary) == n_player
        #         ), f"Should have {n_player} summaries at each round."
        #         for p_summary in summary.values():
        #             for k in ["prompt", "resp"]:
        #                 assert (
        #                     k in p_summary
        #                 ), f"Should have key `{k}` in summary but received {p_summary} in game {req.game_idx}."

        if any(self.use_summary):
            assert (
                len(req.history["summaries"]) == n_rounds
            ), f"Should have same number of summaries as number of rounds in game {req.game_idx}."
            for round_idx, summary in enumerate(req.history["summaries"]):
                for player_idx, p_summary in summary.items():
                    player_model = req.history["models"][int(player_idx)]
                    player_agent = self.agents[player_model]
                    if player_agent.use_summary:
                        for k in ["prompt", "resp"]:
                            assert (
                                k in p_summary
                            ) == player_agent.use_summary, f"Should {'have' if player_agent.use_summary else 'not have'} key `{k}` in summary but received {p_summary} for player {player_idx} in round {round_idx} of game {req.game_idx}."
                    else:
                        assert (
                            p_summary is None
                        ), f"Player {player_idx} should not have a summary in round {round_idx} of game {req.game_idx} as their model doesn't use summaries."

        # role guess check
        # if self.to_guess_role:
        #     assert (
        #         len(req.history["role_guess"]) == n_rounds
        #     ), "Each round should have role guess."
        #     for guess in req.history["role_guess"]:
        #         assert (
        #             len(guess) == n_player - 1
        #         ), f"Should have {n_player-1} guess at each round since Merlin is not included but received {len(guess)} guesses in game {req.game_idx}."
        #         for p_guesses in guess.values():
        #             for p_guess in p_guesses:
        #                 for k in [
        #                     "output",
        #                     "prompt",
        #                     "src_player",
        #                     "tgt_player",
        #                 ]:
        #                     assert (
        #                         k in p_guess
        #                     ), f"Should have key `{k}` in role guess but received {p_guess} in game {req.game_idx}."

        # role belief check
        if self.to_guess_belief:
            assert (
                len(req.history["role_belief"]) == n_rounds
            ), "Each round should have belief guess."
            for guess in req.history["role_belief"]:
                assert (
                    len(guess) == n_player
                ), f"Should have {n_player} belief guess at each round but received {len(guess)} guesses in game {req.game_idx}."
                for p_guesses in guess.values():
                    for p_guess in p_guesses:
                        for k in [
                            "output",
                            "prompt",
                            "src_player",
                            "tgt_player",
                            "tgt_role",
                        ]:
                            assert (
                                k in p_guess
                            ), f"Should have key `{k}` in belief guess but received {p_guess} in game {req.game_idx}."
        self.n_finished_games += 1
        self.logger.info(f"Game {req.game_idx} has finished.")

    def phase_check(self, status, phase):
        # the order might be out of sync
        return
        if (
            status
            in [
                RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
                RequestStatus.TEAM_DISCUSSION_CHECK_ERROR,
                RequestStatus.TEAM_DISCUSSION_SUCCEED,
                RequestStatus.ROLE_GUESS_GET_PROMPT,
                RequestStatus.ROLE_GUESS_CHECK_ERROR,
                RequestStatus.ROLE_GUESS_SUCCEED,
                RequestStatus.ROLE_BELIEF_GET_PROMPT,
                RequestStatus.ROLE_BELIEF_CHECK_ERROR,
                RequestStatus.ROLE_BELIEF_SUCCEED,
                RequestStatus.SUMMARIZE_GET_PROMPT,
                RequestStatus.SUMMARIZE_CHECK_ERROR,
                RequestStatus.SUMMARIZE_SUCCEED,
            ]
            and phase != 0
        ):
            # import pdb

            # pdb.set_trace()
            raise ValueError(
                f"{status} should have phase=0 but received {phase}."
            )

        if (
            status
            in [
                RequestStatus.TEAM_PROPOSAL_GET_PROMPT,
                RequestStatus.TEAM_PROPOSAL_CHECK_ERROR,
                RequestStatus.TEAM_PROPOSAL_SUCCEED,
            ]
            and phase != 0
        ):
            raise ValueError(f"{status} should have phase=0.")

        if (
            status
            in [
                RequestStatus.TEAM_VOTE_GET_PROMPT,
                RequestStatus.TEAM_VOTE_CHECK_ERROR,
                RequestStatus.TEAM_VOTE_SUCCEED,
            ]
            and phase != 1
        ):
            raise ValueError(f"{status} should have phase=1.")

        if (
            status
            in [
                RequestStatus.QUEST_VOTE_GET_PROMPT,
                RequestStatus.QUEST_VOTE_CHECK_ERROR,
                RequestStatus.QUEST_VOTE_SUCCEED,
            ]
            and phase != 2
        ):
            raise ValueError(f"{status} should have phase=2.")

        if (
            status
            in [
                RequestStatus.ASSASSIN_GET_PROMPT,
                RequestStatus.ASSASSIN_CHECK_ERROR,
                RequestStatus.ASSASSIN_VOTE_SUCCEED,
            ]
            and phase != 3
        ):
            raise ValueError(f"{status} should have phase=3.")

    def _add_role_guess_and_belief(self, req, req_queue):
        # if not role guess and role belief, send to summary
        n_player = len(req.history["roles"])
        if self.to_guess_role:
            good_player_ids: List[int] = [
                i
                for i, role in enumerate(req.history["roles"])
                if role[2] and role[1] != "Merlin"
            ]
            for i in good_player_ids:
                for j in range(n_player):
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=i,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.ROLE_GUESS_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                            args={"tgt_player_i": j},
                        ),
                        req_queue=req_queue,
                    )
        elif self.logger:
            self.logger.debug(
                "Role guess is omitted since `to_guess_role` is not specified."
            )

        if self.to_guess_belief:
            for i in range(n_player):
                for j in range(n_player):
                    if i == j:
                        continue
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=i,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                            args={"tgt_player_i": j},
                        ),
                        req_queue=req_queue,
                    )
        elif self.logger:
            self.logger.debug(
                "Role belief is omitted since `to_guess_belief` is not specified."
            )

        if not self.to_guess_role and not self.to_guess_belief:
            self.logger.debug(
                "Since both `role_guess` and `role_belief` are not specified, routing to summarize"
            )
            for i in range(n_player):
                self.process_req(
                    req=Request(
                        prompt=None,
                        resp=None,
                        game_idx=req.game_idx,
                        player_idx=i,
                        history=req.history,
                        env=req.env,
                        status=RequestStatus.SUMMARIZE_GET_PROMPT,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    ),
                    req_queue=req_queue,
                )

    def process_req(
        self,
        req: Request,
        req_queue: List[Request],
    ):
        # (prompt, resp, game idx, history, env, status)
        # All requests in the queue must have prompts except for the first round
        # (i.e. status=0)
        n_round = len(req.history["leaders"])
        assert n_round == req.round_idx, req
        # phase: int = req.env.get_phase()[0]
        # self.phase_check(req.status, phase)
        n_player = self._get_num_players(req)
        if req.player_idx >= len(req.history["models"]):
            self.logger.warning(
                f"req.player_idx={req.player_idx}, req.status={req.status}, models={req.history['models']}"
            )
        current_model = req.history["models"][req.player_idx]
        current_agent = self.agents[current_model]
        use_summary = current_agent.use_summary

        if req.status in [
            RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
            RequestStatus.TEAM_DISCUSSION_CHECK_ERROR,
        ]:
            if not self.to_discuss:
                if self.logger:
                    self.logger.debug(
                        f"`to_discuss` is not specified, routing to next stage. {req}"
                    )
                self._add_role_guess_and_belief(req, req_queue)
                return
            if (
                self.logger
                and req.status == RequestStatus.TEAM_DISCUSSION_GET_PROMPT
            ):
                self.logger.debug(
                    f"Game number: {req.game_idx}.  Round: {n_round}. Team discussion phase, the leader is Player {req.history['leaders'][-1]}, current player is Player {req.player_idx}."
                )
            if len(req.history["team_discs"]) < n_round:
                req.history["team_discs"].append({})
            player_id = req.player_idx
            if player_id in req.history["team_discs"][n_round - 1]:
                raise ValueError(
                    f"status = {req.status} but team discussion is done for game {req.game_idx} round {n_round} player {player_id}."
                )
            prompt, status = current_agent.team_discussion(req)
            if status == RequestStatus.TEAM_DISCUSSION_SUCCEED:
                for i in range(player_id):
                    assert i in req.history["team_discs"][n_round - 1]
                resp = req.resp
                req.history["team_discs"][n_round - 1][player_id] = resp
                if len(req.history["team_discs"][n_round - 1]) == n_player:
                    self._add_role_guess_and_belief(req, req_queue)
                else:
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=req.player_idx + 1,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                        ),
                        req_queue=req_queue,
                    )
                return
            elif status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR:
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return
            else:
                raise ValueError(f"Unknown situation for status={status}.")
        elif req.status in [
            RequestStatus.ROLE_GUESS_GET_PROMPT,
            RequestStatus.ROLE_GUESS_CHECK_ERROR,
        ]:
            role = req.history["roles"][req.player_idx][1]
            assert (
                self.to_guess_role
            ), "Team discussion should not send role guess requests."
            if len(req.history["role_guess"]) < n_round:
                req.history["role_guess"].append({})
            if role == "Merlin" and self.logger:
                self.logger.debug("Merlin is omited in role guessing.")
            if req.status == RequestStatus.ROLE_GUESS_GET_PROMPT:
                self.guess_and_belief_counter[req.game_idx] += 1
                if self.logger:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. Role guess phase. Current player is Player {req.player_idx}. Counter is {self.guess_and_belief_counter[req.game_idx]}"
                    )

            prompt, status = current_agent.guess_role(
                req,
                to_guess_multiple_player=self.to_guess_multiple_player_role,
                use_team_prompt=self.use_team_prompt,
            )
            if status == RequestStatus.ROLE_GUESS_SUCCEED:
                resp = req.resp
                if req.player_idx in req.history["role_guess"][
                    n_round - 1
                ] and any(
                    resp["tgt_player"] == req.buffer["tgt_player_i"]
                    for resp in req.history["role_guess"][n_round - 1][
                        req.player_idx
                    ]
                ):
                    if self.logger:
                        self.logger.debug(
                            f"status = {req.status} but role_guess is done for round={n_round}. This shouldn't happen."
                        )
                        for i, _req in _get_all_prev(req):
                            self.logger.debug(f"{i}: {req}")
                    raise ValueError("Duplicate role geuss request.")
                req.history["role_guess"][n_round - 1].setdefault(
                    req.player_idx, []
                ).append(
                    {
                        "prompt": prompt,
                        "output": resp,
                        "init_resp": req.buffer["init_resp"],
                        "src_player": req.player_idx,
                        "tgt_player": req.buffer["tgt_player_i"],
                        "tgt_role": (
                            req.buffer["tgt_role"]
                            if not self.use_team_prompt
                            else None
                        ),
                    }
                )
                self.guess_and_belief_counter[req.game_idx] -= 1
                if self.logger:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. Role guess succeed. Current player is Player {req.player_idx}. Counter is {self.guess_and_belief_counter[req.game_idx]}"
                    )
                if self.guess_and_belief_counter[req.game_idx] == 0:
                    if self.logger:
                        self.logger.debug(
                            f"Game number: {req.game_idx} has finished all role guess and belief inference. Routing to summarization."
                        )
                    for i in range(n_player):
                        self.process_req(
                            req=Request(
                                prompt=None,
                                resp=None,
                                game_idx=req.game_idx,
                                player_idx=i,
                                history=req.history,
                                env=req.env,
                                status=RequestStatus.SUMMARIZE_GET_PROMPT,
                                prev=None if DEBUG == 0 else deepcopy(req),
                            ),
                            req_queue=req_queue,
                        )
                return
            else:  # get_prompt or error
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return
        elif req.status in [
            RequestStatus.ROLE_BELIEF_GET_PROMPT,
            RequestStatus.ROLE_BELIEF_CHECK_ERROR,
        ]:
            assert (
                self.to_guess_belief
            ), "Team discussion should not send role belief requests."
            if req.status == RequestStatus.ROLE_BELIEF_GET_PROMPT:
                self.guess_and_belief_counter[req.game_idx] += 1
                if self.logger:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. Belief guess phase. Current player is Player {req.player_idx}. Counter is {self.guess_and_belief_counter[req.game_idx]}"
                    )
            if len(req.history["role_belief"]) < n_round:
                req.history["role_belief"].append({})

            prompt, status = current_agent.guess_belief(
                req, use_team_prompt=self.use_team_prompt
            )
            if status == RequestStatus.ROLE_BELIEF_SUCCEED:
                if req.player_idx in req.history["role_belief"][
                    n_round - 1
                ] and any(
                    resp["tgt_player"] == req.buffer["tgt_player_i"]
                    for resp in req.history["role_belief"][n_round - 1][
                        req.player_idx
                    ]
                ):
                    if self.logger:
                        self.logger.debug(
                            f"status = {req.status} but role_belief is done for round={n_round}. This shouldn't happen."
                        )
                        for i, _req in _get_all_prev(req):
                            self.logger.debug(f"{i}: {req}")
                    raise ValueError("Duplicate role belief request.")
                resp = req.resp
                req.history["role_belief"][n_round - 1].setdefault(
                    req.player_idx, []
                ).append(
                    {
                        "prompt": prompt,
                        "output": resp,
                        "init_resp": req.buffer["init_resp"],
                        "src_player": req.player_idx,
                        "tgt_player": req.buffer["tgt_player_i"],
                        "tgt_role": (
                            req.buffer["tgt_role"]
                            if not self.use_team_prompt
                            else None
                        ),
                    }
                )
                self.guess_and_belief_counter[req.game_idx] -= 1
                if self.logger:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. Role belief succeed. Current player is Player {req.player_idx}. Counter is {self.guess_and_belief_counter[req.game_idx]}"
                    )
                if self.guess_and_belief_counter[req.game_idx] == 0:
                    if self.logger:
                        self.logger.debug(
                            f"Game number: {req.game_idx} has finished all role guess and belief inference. Routing to summarization."
                        )
                    for i in range(n_player):
                        self.process_req(
                            req=Request(
                                prompt=None,
                                resp=None,
                                game_idx=req.game_idx,
                                player_idx=i,
                                history=req.history,
                                env=req.env,
                                status=RequestStatus.SUMMARIZE_GET_PROMPT,
                                prev=None if DEBUG == 0 else deepcopy(req),
                            ),
                            req_queue=req_queue,
                        )
                return
            else:  # get_prompt or error
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return
        elif req.status in [
            RequestStatus.SUMMARIZE_GET_PROMPT,
            RequestStatus.SUMMARIZE_CHECK_ERROR,
        ]:
            if not use_summary:
                self.n_summaries[req.game_idx] += 1
                if self.n_summaries[req.game_idx] == n_player:
                    # if DEBUG in [1, 2]:
                    #     filtered_reqs = [
                    #         _req
                    #         for _req in self.cur_queue
                    #         if _req.game_idx == req.game_idx
                    #         and _req.player_idx == req.player_idx
                    #     ]
                    #     if len(filtered_reqs) != 0:
                    #         self.logger.debug(
                    #             f"Have {len(filtered_reqs)} reqs for {req.game_idx} at summary stage. {req}"
                    #         )
                    #         self.logger.debug(
                    #             "\n\n".join(
                    #                 [str(_req) for _req in filtered_reqs]
                    #             )
                    #         )
                    #         raise ValueError()
                    if self.logger:
                        self.logger.debug(
                            f"`use_summary` is not specified. {self.n_summaries[req.game_idx]} players reached this round so we route request to `team_proposal`. {req}"
                        )
                    self.n_summaries[req.game_idx] = 0
                    leader: int = req.history["leaders"][-1]
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=leader,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.TEAM_PROPOSAL_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                        ),
                        req_queue=req_queue,
                    )
                else:
                    if self.logger:
                        self.logger.debug(
                            f"`use_summary` is not specified. {self.n_summaries[req.game_idx]} players reached only so we will wait. {req}"
                        )
                return
            if (
                self.logger
                and req.status == RequestStatus.SUMMARIZE_GET_PROMPT
            ):
                self.logger.debug(
                    f"Game number: {req.game_idx}.  Round: {n_round}. Summarize phase. Current player is Player {req.player_idx}."
                )
            if len(req.history["summaries"]) < n_round:
                req.history["summaries"].append({})
            if req.player_idx in req.history["summaries"][n_round - 1]:
                raise ValueError(
                    f"status = {req.status} but summarization is done for game {req.game_idx} round {n_round-1}."
                )

            prompt, status = current_agent.summarize(req)
            if status == RequestStatus.SUMMARIZE_SUCCEED:
                resp = req.resp
                req.history["summaries"][n_round - 1][req.player_idx] = {
                    "prompt": prompt,
                    "resp": resp,
                    "init_resp": req.buffer["init_resp"],
                }
                # need to wait all finish summarizing.
                n_total_summaries = sum(
                    self.agents[model].use_summary
                    for model in req.history["models"]
                )
                if (
                    len(req.history["summaries"][n_round - 1])
                    == n_total_summaries
                ):
                    leader: int = req.history["leaders"][-1]
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=leader,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.TEAM_PROPOSAL_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                        ),
                        req_queue=req_queue,
                    )
                return
            else:
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return
        elif req.status in [
            RequestStatus.TEAM_PROPOSAL_GET_PROMPT,
            RequestStatus.TEAM_PROPOSAL_CHECK_ERROR,
        ]:
            if n_round == len(req.history["team_props"]):
                if self.logger:
                    self.logger.debug(
                        f"The team {req.history['team_props'][n_round-1]['team']} has been proposed for round {n_round}. This request will be omitted."
                    )
                return

            leader: int = req.history["leaders"][-1]
            if req.player_idx != leader:
                if self.logger:
                    self.logger.debug(
                        f"Player {req.player_idx} is not the leader and is thus omitted in team_proposal."
                    )
                return
            if self.logger:
                self.logger.debug(
                    f"Game number: {req.game_idx}. Round: {n_round}. Team proposal phase. Current player is Player {req.player_idx}."
                )
            prompt, status = current_agent.propose_team(req)
            if status == RequestStatus.TEAM_PROPOSAL_SUCCEED:
                resp: Dict[str, str] = req.resp
                if self.logger:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. Team selection phase, the leader, Player {leader}, selected the team {resp['team']}."
                    )
                req.history["team_props"].append(resp)
                req.env.choose_quest_team(
                    team=frozenset(resp["team"]), leader=leader
                )
                for i in range(n_player):
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=i,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.TEAM_VOTE_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                        ),
                        req_queue=req_queue,
                    )
                return
            else:
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return
        elif req.status in [
            RequestStatus.TEAM_VOTE_GET_PROMPT,
            RequestStatus.TEAM_VOTE_CHECK_ERROR,
        ]:
            # history["team_votes"][0] = {"votes": Dict[player_i -> {"rationale", "vote": T/F, "prompt"}], "result": T/F}
            if len(req.history["team_votes"]) < n_round:
                req.history["team_votes"].append({"votes": {}, "result": None})

            if self.logger:
                leader: int = req.history["leaders"][-1]
                team = req.env.get_current_quest_team()
                if req.status == RequestStatus.TEAM_VOTE_GET_PROMPT:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. Team vote phase, Player {req.player_idx} voting on team {team} chosen by {leader}."
                    )
            prompt, status = current_agent.vote_on_team(req)
            if status == RequestStatus.TEAM_VOTE_SUCCEED:
                # `rationale` (str): The rationale for the vote.
                # `vote` (bool): The outcome of the vote
                #     (True for "approve", False for "reject").
                resp: Dict[str, str] = req.resp
                req.history["team_votes"][-1]["votes"][req.player_idx] = resp
                # need to wait all five
                if not len(req.history["team_votes"][-1]["votes"]) == n_player:
                    return
                votes = [
                    vote["vote"]
                    for vote in req.history["team_votes"][-1]["votes"].values()
                ]
                result = req.env.gather_team_votes(votes)
                req.history["team_votes"][-1]["result"] = result[-1]
                approved_votes = sum(votes)
                if self.logger:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. {approved_votes} approved, {len(votes) - approved_votes} failed. The team is {'accepted' if result[-1] else 'failed'}."
                    )

                phase: int = req.env.get_phase()[0]
                if phase == 2:
                    for i in range(n_player):
                        self.process_req(
                            req=Request(
                                prompt=None,
                                resp=None,
                                game_idx=req.game_idx,
                                player_idx=i,
                                history=req.history,
                                env=req.env,
                                status=RequestStatus.QUEST_VOTE_GET_PROMPT,
                                prev=None if DEBUG == 0 else deepcopy(req),
                            ),
                            req_queue=req_queue,
                        )
                    return
                elif phase == 0:
                    # back to discuss
                    req.history["leaders"].append(req.env.get_quest_leader())
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=0,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                        ),
                        req_queue=req_queue,
                    )
                    return
                else:
                    raise ValueError(f"Unknown phase {phase}.")
            else:
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return
        elif req.status in [
            RequestStatus.QUEST_VOTE_GET_PROMPT,
            RequestStatus.QUEST_VOTE_CHECK_ERROR,
        ]:
            if len(req.history["quest_votes"]) < n_round:
                req.history["quest_votes"].append(
                    {"votes": {}, "result": None}
                )
            quest_team = req.env.get_current_quest_team()
            if req.player_idx not in quest_team:
                if self.logger:
                    self.logger.debug(
                        f"Game number: {req.game_idx}.  Round: {n_round}. {req.player_idx} not in quest team, so this request is ignored. "
                    )
                return

            if self.logger:
                self.logger.debug(
                    f"Game number: {req.game_idx}.  Round: {n_round}. Quest vote phase, Player {req.player_idx} voting."
                )
            prompt, status = current_agent.vote_on_mission(req)
            if status == RequestStatus.QUEST_VOTE_SUCCEED:
                # `rationale` (str): The rationale for the vote.
                # `vote` (bool): The outcome of the vote
                #     (True for "approve", False for "reject").
                resp: Dict[str, str] = req.resp
                req.history["quest_votes"][-1]["votes"][req.player_idx] = resp
                # need to wait all votes
                if not len(req.history["quest_votes"][-1]["votes"]) == len(
                    quest_team
                ):
                    return
                votes = [
                    vote["vote"]
                    for vote in req.history["quest_votes"][-1][
                        "votes"
                    ].values()
                ]
                result = req.env.gather_quest_votes(votes)
                req.history["quest_votes"][-1]["result"] = result[-2]
                num_failed = result[-1]
                if self.logger:
                    self.logger.debug(
                        f"{len(votes) - num_failed} approved, {num_failed} failed. The quest {'suceeds' if result[-2] else 'fails'}."
                    )

                phase: int = req.env.get_phase()[0]
                if phase == 3:
                    assassin = req.env.get_assassin()
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=assassin,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.ASSASSIN_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                        ),
                        req_queue=req_queue,
                    )
                    return
                elif phase == 0:
                    # back to discuss
                    req.history["leaders"].append(req.env.get_quest_leader())
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=0,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
                            prev=None if DEBUG == 0 else deepcopy(req),
                        ),
                        req_queue=req_queue,
                    )
                    return
                elif req.env.done:
                    req.history["final_result"] = req.env.good_victory
                    req.history["status"] = "Finished"
                    self.finish_game(req)
                    return
                else:
                    raise ValueError(f"Unknown phase {phase}.")
            else:
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return

        elif req.status in [
            RequestStatus.ASSASSIN_GET_PROMPT,
            RequestStatus.ASSASSIN_CHECK_ERROR,
        ]:

            if self.logger:
                self.logger.debug(
                    f"Game number: {req.game_idx}.  Round: {n_round}. Assassin phase, Player {req.player_idx} is choosing."
                )
            prompt, status = current_agent.assassinate(req)
            if status == RequestStatus.ASSASSIN_VOTE_SUCCEED:
                resp: Dict[str, str] = req.resp
                # (next phase, game is done, good wins?)
                assassin = resp["merlin"]
                result = req.env.choose_assassination_target(
                    req.player_idx, resp["merlin"]
                )
                resp["success"] = not result[-1]
                req.history["assassin"] = resp
                req.history["final_result"] = req.env.good_victory
                req.history["status"] = "Finished"
                if self.logger:
                    self.logger.debug(
                        f"The assassination is {'failed' if result[-1] else 'successful'}."
                    )
                self.finish_game(req)
                return
            else:
                req_queue.append(
                    Request(
                        prompt=prompt,
                        resp=req.resp,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=status,
                        buffer=req.buffer,
                        prev=None if DEBUG == 0 else deepcopy(req),
                    )
                )
                return
        else:
            raise ValueError(f"Unknown status {req.status}")
            # history["team_votes"][0] = {"votes": Dict[player_i -> {"rationale", "vote": T/F, "prompt"}], "result": T/F}


def main(
    output_path,
    model_name=None,
    model_names: List[
        Dict[str, Union[int, str]],
    ] = None,  # (model_name, model_path, port number)
    lora_names: List[
        Dict[str, Union[int, str]],
    ] = None,  # (model_name, model_path, lora_path)
    n_games=20,
    game_batch_size: int = None,
    inference_strategy: str = "vllm",
    seed: int = 111,
    max_tokens: int = 512,
    temperature=0,
    top_p=1,
    to_exchange_role_per_setting: bool = True,
    to_discuss=True,
    add_strategy_in_prompt=False,
    add_quest_strategy_in_prompt=False,
    to_guess_role: bool = False,
    to_guess_multiple_player_role: bool = False,
    to_guess_belief: bool = False,
    use_summary: Union[bool, List[bool]] = False,
    include_prev_disc: Union[bool, List[bool]] = True,
    n_gpus: int = 1,
    seed_global: bool = False,
    use_team_prompt: bool = False,
):
    """_summary_

    Args:
        model_name (_type_): _description_
        output_path (_type_): _description_
        model_names (_type_, optional):
            (model_name, model_path, port number)
        seed (int, optional): _description_. Defaults to 111.
        max_tokens (int, optional): _description_. Defaults to 512.
        temperature (int, optional): _description_. Defaults to 0.
        top_p (int, optional): _description_. Defaults to 1.
        to_exchange_role_per_setting: If True, we arrange two games per setting
             where the two models played against different teams.
        to_discuss (bool, optional): _description_. Defaults to True.
        add_strategy_in_prompt (bool, optional): _description_. Defaults to False.
        to_guess_role (bool, optional): _description_. Defaults to False.
        to_guess_multiple_player_role (bool, optional): _description_. Defaults to False.
        to_guess_belief (bool, optional): _description_. Defaults to False.
        use_summary (bool, optional): _description_. Defaults to False.
        n_gpus (int, optional): _description_. Defaults to 1.
        use_team_prompt: use team prompt in role guess and role belief.

    Raises:
        ValueError: _description_
    """
    seeder = random.Random(seed)
    if game_batch_size is None:
        game_batch_size = n_games
    if model_names is not None:
        # assert len(model_names) == 2, "Should provide two models."
        for _model in model_names:
            for k in ["port", "path", "name"]:
                assert (
                    k in _model
                ), f"The key `{k}` should be provided. Received: {_model}"
    if lora_names is not None:
        for _model in lora_names:
            for k in ["lora_path", "path", "name"]:
                assert (
                    k in _model
                ), f"The key `{k}` should be provided. Received: {_model}"
    assert (
        use_team_prompt
    ), "`use_team_prompt` can only be true since I didn't support it in `_add_role_guess_and_belief`."
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_name = os.path.split(output_path)[-1].rsplit(".", 1)[0] + ".txt"
    logger = get_logger(
        __name__,
        logger_level="debug",
        console_level="debug",
        file_level="debug",
        log_path=os.path.join("logs", log_name),
    )
    if (model_name is not None) + (model_names is not None) + (
        lora_names is not None
    ) != 1:
        raise ValueError(
            "Either provide `model_name` or `model_names` or `lora_names`."
        )

    if to_guess_role != to_guess_belief:
        raise ValueError(
            "Current only support guess role and guess belief together."
        )

    # init
    reqs = []
    if DEBUG == 1 and inference_strategy == "vllm" and model_names is not None:
        # just use one model here for debug
        from vllm import LLM, SamplingParams

        _model_name = model_names[0]["name"]
        model_path = MODELS[_model_name]["path"]
        model = LLM(
            model=model_path,
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
    if DEBUG == 1 or inference_strategy == "local":
        if model_name is not None:
            # model = VllmWrapper(
            #     strategy=AnyscaleInferenceStrategy(), model_name=model_name
            # )
            model_path = MODELS[model_name]["path"]
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = VllmStrategyWrapper(
                strategy=LocalInferenceStrategy(
                    model=model,
                    tokenizer=tokenizer,
                    chat_template=get_conv_template("llama-3"),
                ),
                model_name=model_path,
                end_tokens=[tokenizer.eos_token],
            )
        else:
            logger.warning(
                "Using multiple models with local models are supposed only for debugging purpose."
            )
            if DEBUG == 0:
                raise ValueError(
                    "Using multiple models with local models are only allowed with `DEBUG`."
                )
            models = {}
            for i, model_config in enumerate(model_names):
                model_path = model_config["path"]
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = VllmStrategyWrapper(
                    strategy=LocalInferenceStrategy(
                        model=model,
                        tokenizer=tokenizer,
                        chat_template=get_conv_template(
                            MODELS[model_config["name"]]["template"]
                        ),
                    ),
                    model_name=model_path,  # doesn't matter
                    end_tokens=[tokenizer.eos_token],
                )
                models[model_config["name"]] = model
        sampling_params = SamplingParamsWrapper(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
    elif inference_strategy == "openai":
        model = VllmStrategyWrapper(
            strategy=OpenAIInferenceStrategy(),
            model_name=model_name,
            end_tokens=[tokenizer.eos_token],
        )
        sampling_params = SamplingParamsWrapper(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
    elif inference_strategy == "together":
        model = VllmStrategyWrapper(
            strategy=TogetherInferenceStrategy(),
            model_name=model_name,
            end_tokens=[tokenizer.eos_token],
        )
        sampling_params = SamplingParamsWrapper(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
    elif inference_strategy == "anyscale":
        model = VllmStrategyWrapper(
            strategy=AnyscaleInferenceStrategy(),
            model_name=model_name,
            end_tokens=[tokenizer.eos_token],
        )
        sampling_params = SamplingParamsWrapper(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
    elif inference_strategy == "vllm" and model_name is not None:
        from vllm import LLM, SamplingParams

        model_path = MODELS[model_name]["path"]
        model = LLM(
            model=model_path,
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
    elif inference_strategy == "vllm" and lora_names is not None:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        paths = list(set([n["path"] for n in lora_names]))
        assert (
            len(paths) == 1
        ), f"Only one base model is currently allowed for lora. Received: {paths}"
        model = LLM(
            model=paths[0],
            dtype="float16",
            tensor_parallel_size=n_gpus,
            seed=seed if seed_global else 0,
            enable_lora=True,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed if not seed_global else None,
        )
    else:
        assert (
            inference_strategy == "vllm"
        ), "Multiple models are only avilable with vllm."

    agents = []
    if model_name is not None:
        agent = VllmAgent(
            add_strategy_in_prompt=add_strategy_in_prompt,
            add_quest_strategy_in_prompt=add_quest_strategy_in_prompt,
            use_summary=use_summary,
            include_prev_disc=include_prev_disc,
            chat_template=get_conv_template(MODELS[model_name]["template"]),
        )
        agents.append((model_name, agent))
    else:
        _model_names = model_names if model_names is not None else lora_names
        for i, model_config in enumerate(_model_names):
            agent = VllmAgent(
                add_strategy_in_prompt=add_strategy_in_prompt,
                add_quest_strategy_in_prompt=add_quest_strategy_in_prompt,
                use_summary=(
                    use_summary[i]
                    if isinstance(use_summary, list)
                    else use_summary
                ),
                include_prev_disc=(
                    include_prev_disc[i]
                    if isinstance(include_prev_disc, list)
                    else include_prev_disc
                ),
                chat_template=get_conv_template(
                    MODELS[model_config["name"]]["template"]
                ),
            )
            agents.append((model_config["name"], agent))

    req_processor = RequestProcessor(
        agents=agents,
        to_discuss=to_discuss,
        add_strategy_in_prompt=add_strategy_in_prompt,
        to_guess_role=to_guess_role,
        to_guess_multiple_player_role=to_guess_multiple_player_role,
        n_guess_role_repeat=1,
        to_guess_belief=to_guess_belief,
        use_summary=use_summary,
        logger=logger,
        use_team_prompt=use_team_prompt,
    )
    histories = []
    n_players = 6
    config = AvalonBasicConfig.from_num_players(
        n_players, percival=True, morgana=True
    )
    req_round_i = 0
    # n_game = 50, game_batch_size = 20,
    # range(1, 21), range(21, 41), range(41, 51)
    for start_game_i in range(1, n_games + 1, game_batch_size):
        end_game_i = min(start_game_i + game_batch_size, n_games + 1)
        for game_i in range(start_game_i, end_game_i):
            env = AvalonGameEnvironment(config, seed=seed + game_i)
            history = {
                "leaders": [env.get_quest_leader()],
                "team_discs": [],
                "team_props": [],
                "team_votes": [],
                "quest_votes": [],
                "role_guess": [],
                "role_belief": [],
                "summaries": [],
                "assassin": None,
                "roles": [
                    (int(role_tuple[0]), role_tuple[1], bool(role_tuple[2]))
                    for role_tuple in env.get_roles()
                ],
                "input_tokens": 0,
                "output_tokens": 0,
                "n_error": 0,
                "id": game_i,
            }
            if model_names is not None or lora_names is not None:
                if model_names is not None:
                    current_model_names = [m["name"] for m in model_names]
                elif lora_names is not None:
                    current_model_names = [m["name"] for m in lora_names]

                if len(current_model_names) > 2:
                    # allow duplicate here
                    current_model_names = seeder.choices(
                        current_model_names, k=2
                    )
                    # current_model_names = seeder.sample(
                    #     current_model_names, k=2
                    # )

                new_history = construct_two_players_game(
                    history,
                    current_model_names,
                )
                reqs.append(
                    Request(
                        prompt=None,
                        resp=None,
                        game_idx=len(reqs) + 1,
                        player_idx=0,
                        history=new_history,
                        env=env,
                        status=RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
                    )
                )
                histories.append(new_history)

                # game 2
                if to_exchange_role_per_setting:
                    env = deepcopy(env)
                    new_history = deepcopy(history)
                    new_history = construct_two_players_game(
                        history, current_model_names, revert=True
                    )
                    reqs.append(
                        Request(
                            prompt=None,
                            resp=None,
                            game_idx=len(reqs) + 1,
                            player_idx=0,
                            history=new_history,
                            env=env,
                            status=RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
                        )
                    )
                    histories.append(new_history)
            else:
                history["models"] = [model_name] * n_players
                history["n_error"] = {model_name: 0}
                histories.append(history)
                # (prompt, resp, game idx, history, env, status, buffer)
                # buffer mainly for storing temporary messages.
                # The status code performs error checking and processing.
                reqs.append(
                    Request(
                        prompt=None,
                        resp=None,
                        game_idx=game_i,
                        player_idx=0,
                        history=history,
                        env=env,
                        status=RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
                    )
                )

        # sample until finish
        if model_names is not None:
            pbar = tqdm(total=len(reqs), desc="Sampling games")
        else:
            pbar = tqdm(total=len(reqs), desc="Sampling games")
        count = 0
        while reqs:
            if DEBUG == 2:
                with open(f"outputs/reqs_{req_round_i}.json", "w") as f:
                    json.dump([req.to_dict() for req in reqs], f, indent=4)
            req_round_i += 1
            new_reqs = []
            for req in reqs:
                if DEBUG in [1, 2]:
                    req_processor._set_curr_queue(reqs)
                req_processor.process_req(
                    req=req,
                    req_queue=new_reqs,
                )
            reqs = new_reqs
            if model_name is not None or (
                model_names is not None
                and DEBUG in [1, 2]
                and inference_strategy == "vllm"
            ):
                resps = model.generate(
                    [req.prompt for req in reqs],
                    sampling_params=sampling_params,
                    # use_tqdm=False,
                )
            else:  # multiple models
                if DEBUG == 1:
                    resps = []
                    for req in reqs:
                        model_name = req.history["models"][req.player_idx]
                        resp = models[model_name].generate(
                            [req.prompt],
                            sampling_params=sampling_params,
                            # use_tqdm=False,
                        )
                        resps += resp
                elif lora_names is not None:
                    idx2resp = {}
                    for j, model_config in enumerate(lora_names):
                        idxes = []
                        curr_reqs = []
                        for i, req in enumerate(reqs):
                            if (
                                req.history["models"][req.player_idx]
                                == model_config["name"]
                            ):
                                curr_reqs.append(req)
                                idxes.append(i)
                        resps = model.generate(
                            [req.prompt for req in curr_reqs],
                            sampling_params=sampling_params,
                            lora_request=(
                                LoRARequest(
                                    model_config["name"],
                                    j,
                                    model_config["lora_path"],
                                )
                                if model_config["lora_path"]
                                else None
                            ),
                            # use_tqdm=False,
                        )
                        for idx, resp in zip(idxes, resps):
                            idx2resp[idx] = resp
                    assert len(idx2resp) == len(reqs)
                    resps = [idx2resp[i] for i in range(len(resps))]
                else:
                    args = []
                    for req in reqs:
                        for model_i in range(len(model_names)):
                            model_path = model_names[model_i]["path"]
                            if (
                                req.history["models"][req.player_idx]
                                == model_names[model_i]["name"]
                            ):
                                args.append(
                                    (
                                        (
                                            req.buffer["msg"]
                                            if USE_MESSAGE
                                            else req.prompt
                                        ),
                                        model_path,
                                        max_tokens,
                                        temperature,
                                        top_p,
                                        f"http://localhost:{model_names[model_i]['port']}/v1",
                                    )
                                )
                                break
                    assert len(args) == len(reqs)
                    p = Pool(500)
                    outputs = p.starmap(wrapper, args)
                    resps = [
                        RequestOutput(
                            prompt=req.prompt,
                            outputs=[output["output"]],
                            prompt_len=output["prompt_len"],
                            output_len=output["output_len"],
                        )
                        for req, output in zip(reqs, outputs)
                    ]
                    p.close()

            for req, resp in zip(reqs, resps):
                req.resp = resp.outputs[0].text
                if (
                    model_name is not None or lora_names is not None
                ) and inference_strategy == "vllm":
                    req.history["input_tokens"] += len(resp.prompt_token_ids)
                    req.history["output_tokens"] += len(
                        resp.outputs[0].token_ids
                    )
                else:
                    req.history["input_tokens"] += resp.prompt_len
                    req.history["output_tokens"] += resp.output_len
            pbar.n = req_processor.n_finished_games
            pbar.refresh()
            count += 1

            if count % 5 == 0:
                with open(
                    output_path, "w", encoding="utf-8", errors="replace"
                ) as f:
                    f.write(
                        "\n".join(
                            [
                                json.dumps(row, ensure_ascii=False)
                                for row in histories
                            ]
                        )
                    )
                logger.info(f"Games saved to {output_path}.")
            # debug
            # with open("outputs/reqs.pkl", "wb") as f:
            #     pickle.dump(reqs, f)
            # debug

    # save all
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in histories]
            )
        )
    logger.info(f"{len(histories)} games saved to {output_path}.")


if __name__ == "__main__":
    StrictFire(main)
