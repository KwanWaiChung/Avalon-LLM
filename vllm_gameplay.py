import json
import random
import os
import pickle
from src.server.tasks.avalon.engine import AvalonGameEnvironment
from src.utils.vllm_misc import Request, RequestStatus
from src.server.tasks.avalon.agents.my_vllm_agent import VllmAgent
from src.utils.inference import (
    DummyInferenceStrategy,
    AnyscaleInferenceStrategy,
)
from fastchat.conversation import get_conv_template
from src.utils.logger import get_logger
from typing import List, Dict, Union
from strictfire import StrictFire
from tqdm import tqdm


DEBUG = True

if not DEBUG:
    from vllm import LLM, SamplingParams


class Output:
    def __init__(self, text: str):
        self.text = text


class RequestOutput:
    def __init__(self, prompt: str, outputs: List[str]):
        self.prompt = prompt
        self.outputs = [Output(text=o) for o in outputs]


class SamplingParamsDummy:
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


class VllmWrapper:
    # this class wraps local inference pretending to be vllm
    def __init__(self, strategy, model_name):
        self.strategy = strategy
        self.model_name = model_name

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
            )
            outputs.append(
                RequestOutput(prompt=prompt, outputs=[output["output"]])
            )
        return outputs


class RequestProcessor:
    def __init__(
        self,
        agent: VllmAgent,
        to_discuss: bool,  # game arguments
        add_strategy_in_history: bool,
        to_guess_role: bool,
        to_guess_multiple_player_role: bool,
        n_guess_role_repeat: int,
        to_guess_belief: bool,
        use_summary: bool,
        logger=None,
    ):
        self.agent = agent
        self.to_discuss = to_discuss
        self.add_strategy_in_history = add_strategy_in_history
        self.to_guess_role = to_guess_role
        self.to_guess_multiple_player_role = to_guess_multiple_player_role
        self.n_guess_role_repeat = n_guess_role_repeat
        self.to_guess_belief = to_guess_belief
        self.use_summary = use_summary
        self.logger = logger
        self.n_finished_games = 0

    def finish_game(self, req):
        if not req.env.done:
            raise ValueError("Game not yet done")

        n_rounds = len(req.history["leaders"])
        # team discs check
        if self.to_discuss:
            assert len(req.history["team_discs"]) == n_rounds
            for team_disc in req.history["team_discs"]:
                assert len(team_disc) == 5, "Should have 5 players discussed"
                for player in team_disc.values():
                    for k in ["strategy", "response", "prompt"]:
                        assert k in player

        # team props check
        assert len(req.history["team_props"]) == n_rounds
        for team_prop in req.history["team_props"]:
            for k in ["rationale", "team", "prompt"]:
                assert k in team_prop, f"{k} should be in `team_prop`."

        # team_votes check
        for team_vote in req.history["team_votes"]:
            assert len(team_vote["votes"]) == 5, "Should have 5 votes"

        # quest vote check
        success_teams = sum([v["result"] for v in req.history["team_votes"]])
        assert (
            sum([v["result"] is not None for v in req.history["quest_votes"]])
            == success_teams
        ), "Number of non-none quest votes should equal to number of sucessed team."

        # summaries check
        if self.use_summary:
            assert (
                len(req.history["summaries"]) == n_rounds
            ), "Should have same number of summaries as number of rounds."
            for summary in req.history["summaries"]:
                assert (
                    len(summary) == 5
                ), "Should have 5 summaries at each round."
                for p_summary in summary.values():
                    for k in ["prompt", "resp"]:
                        assert (
                            k in p_summary
                        ), f"Should have key `{k} in summary."

        # role guess check
        if self.to_guess_role:
            assert (
                len(req.history["role_guess"]) == n_rounds
            ), "Each round should have role guess."
            for guess in req.history["role_guess"]:
                assert (
                    len(guess) == 4
                ), "Should have 4 guess at each round since Merlin is not included."
                for p_guess in guess.values():
                    for k in ["output", "prompt", "src_player", "tgt_player"]:
                        assert (
                            k in p_guess
                        ), f"Should have key `{k} in role guess."

        # role belief check
        if self.to_guess_belief:
            assert (
                len(req.history["role_belief"]) == n_rounds
            ), "Each round should have belief guess."
            for guess in req.history["role_belief"]:
                assert (
                    len(guess) == 5
                ), "Should have 5 belief guess at each round."
                for p_guess in guess.values():
                    for k in [
                        "output",
                        "prompt",
                        "src_player",
                        "tgt_player",
                        "tgt_role",
                    ]:
                        assert (
                            k in p_guess
                        ), f"Should have key `{k} in belief guess."
        self.n_finished_games += 1

    def phase_check(self, status, phase):
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
            raise ValueError(f"{status} should have phase=0.")

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

    def process_req(
        self,
        req: Request,
        req_queue: List[Request],
    ):
        # (prompt, resp, game idx, history, env, status)
        # All requests in the queue must have prompts except for the first round
        # (i.e. status=0)
        phase: int = req.env.get_phase()[0]
        self.phase_check(req.status, phase)
        n_round = len(req.history["leaders"])

        if req.status in [
            RequestStatus.TEAM_DISCUSSION_GET_PROMPT,
            RequestStatus.TEAM_DISCUSSION_CHECK_ERROR,
        ]:
            if self.logger:
                self.logger.info(
                    f"Game number: {req.game_idx}. Team selection phase, the leader is Player {req.history['leaders'][-1]}."
                )
            if not self.to_discuss:
                if self.logger:
                    self.logger.debug(
                        f"`to_discuss` is not specified, routing request to `guess_role`. {req}"
                    )
                for i in range(5):
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=req.player_idx,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.ROLE_GUESS_GET_PROMPT,
                        ),
                        req_queue=req_queue,
                    )
                return
            if len(req.history["team_discs"]) < n_round:
                req.history["team_discs"].append({})
            player_id = req.player_idx
            if player_id in req.history["team_discs"][n_round - 1]:
                raise ValueError(
                    f"status = {req.status} but team discussion is done."
                )
            prompt, status = self.agent.team_discussion(req)
            if status == RequestStatus.TEAM_DISCUSSION_SUCCEED:
                resp = req.resp
                resp["prompt"] = prompt
                req.history["team_discs"][n_round - 1][player_id] = resp
                if req.player_idx == 4:
                    for i in range(5):
                        self.process_req(
                            req=Request(
                                prompt=None,
                                resp=None,
                                game_idx=req.game_idx,
                                player_idx=i,
                                history=req.history,
                                env=req.env,
                                status=RequestStatus.ROLE_GUESS_GET_PROMPT,
                            ),
                            req_queue=req_queue,
                        )
                    return
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
                        ),
                        req_queue=req_queue,
                    )
                    return
            elif status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR:
                # status code must be 1
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
                    )
                )
                return
            else:
                raise ValueError(f"Unknown situation for status={status}.")
        elif req.status in [
            RequestStatus.ROLE_GUESS_GET_PROMPT,
            RequestStatus.ROLE_GUESS_CHECK_ERROR,
        ]:
            role = req.env.get_role(req.player_idx)[1]
            if not self.to_guess_role or role == "Merlin":
                if not self.to_guess_role:
                    if self.logger:
                        self.logger.debug(
                            f"`to_guess_role` is not specified, routing request to `guess_belief`. {req}"
                        )
                elif self.logger:
                    self.logger.debug("Merlin is omited in role guessing.")
                self.process_req(
                    req=Request(
                        prompt=None,
                        resp=None,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
                    ),
                    req_queue=req_queue,
                )
                return
            if len(req.history["role_guess"]) < n_round:
                req.history["role_guess"].append({})
            if req.player_idx in req.history["role_guess"][n_round - 1]:
                raise ValueError(
                    f"status = {req.status} but role_guess is done."
                )

            prompt, status = self.agent.guess_role(
                req,
                to_guess_multiple_player=self.to_guess_multiple_player_role,
            )
            if status == RequestStatus.ROLE_GUESS_SUCCEED:
                resp = req.resp
                req.history["role_guess"][n_round - 1][req.player_idx] = {
                    "prompt": prompt,
                    "output": resp,
                    "src_player": req.player_idx,
                    "tgt_player": req.buffer["tgt_player_i"],
                }
                self.process_req(
                    req=Request(
                        prompt=None,
                        resp=None,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=RequestStatus.ROLE_BELIEF_GET_PROMPT,
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
                    )
                )
                return
        elif req.status in [
            RequestStatus.ROLE_BELIEF_GET_PROMPT,
            RequestStatus.ROLE_BELIEF_CHECK_ERROR,
        ]:
            if not self.to_guess_belief:
                if self.logger:
                    self.logger.debug(
                        f"`to_guess_belief` is not specified, routing request to `summarize`. {req}"
                    )
                self.process_req(
                    req=Request(
                        prompt=None,
                        resp=None,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=RequestStatus.SUMMARIZE_GET_PROMPT,
                    ),
                    req_queue=req_queue,
                )
                return
            if len(req.history["role_belief"]) < n_round:
                req.history["role_belief"].append({})

            if req.player_idx in req.history["role_belief"][n_round - 1]:
                raise ValueError(
                    f"status = {req.status} but role_belief is done."
                )

            prompt, status = self.agent.guess_belief(req)
            if status == RequestStatus.ROLE_BELIEF_SUCCEED:
                resp = req.resp
                req.history["role_belief"][n_round - 1][req.player_idx] = {
                    "prompt": prompt,
                    "output": resp,
                    "src_player": req.player_idx,
                    "tgt_player": req.buffer["tgt_player_i"],
                    "tgt_role": req.buffer["tgt_role"],
                }
                self.process_req(
                    req=Request(
                        prompt=None,
                        resp=None,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=RequestStatus.SUMMARIZE_GET_PROMPT,
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
                    )
                )
                return
        elif req.status in [
            RequestStatus.SUMMARIZE_GET_PROMPT,
            RequestStatus.SUMMARIZE_CHECK_ERROR,
        ]:
            if not self.use_summary:
                if self.logger:
                    self.logger.debug(
                        f"`use_summary` is not specified, routing request to `team_proposal`. {req}"
                    )
                self.process_req(
                    req=Request(
                        prompt=None,
                        resp=None,
                        game_idx=req.game_idx,
                        player_idx=req.player_idx,
                        history=req.history,
                        env=req.env,
                        status=RequestStatus.TEAM_PROPOSAL_GET_PROMPT,
                    ),
                    req_queue=req_queue,
                )
                return
            if len(req.history["summaries"]) < n_round:
                req.history["summaries"].append({})
            if req.player_idx in req.history["summaries"][n_round - 1]:
                raise ValueError(
                    f"status = {req.status} but summarize is done."
                )

            prompt, status = self.agent.summarize(req)
            if status == RequestStatus.SUMMARIZE_SUCCEED:
                resp = req.resp
                req.history["summaries"][n_round - 1][req.player_idx] = {
                    "prompt": prompt,
                    "resp": resp,
                }
                # need to wait all finish summarizing.
                if len(req.history["summaries"][n_round - 1]) == 5:
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
                    )
                )
                return
        elif req.status in [
            RequestStatus.TEAM_PROPOSAL_GET_PROMPT,
            RequestStatus.TEAM_PROPOSAL_CHECK_ERROR,
        ]:
            leader: int = req.history["leaders"][-1]
            if req.player_idx != leader:
                if self.logger:
                    self.logger.debug(
                        f"Player {req.player_idx} is not the leader and is thus omitted in team_proposal."
                    )
                return
            prompt, status = self.agent.propose_team(req)
            if status == RequestStatus.TEAM_PROPOSAL_SUCCEED:
                resp: Dict[str, str] = req.resp
                if self.logger:
                    self.logger.info(
                        f"Game number: {req.game_idx}. Round: {req.env.turn+1}. Team selection phase, the leader, Player {leader}, selected the team {resp['team']}."
                    )
                resp["prompt"] = prompt
                req.history["team_props"].append(resp)
                req.env.choose_quest_team(
                    team=frozenset(resp["team"]), leader=leader
                )
                for i in range(5):
                    self.process_req(
                        req=Request(
                            prompt=None,
                            resp=None,
                            game_idx=req.game_idx,
                            player_idx=i,
                            history=req.history,
                            env=req.env,
                            status=RequestStatus.TEAM_VOTE_GET_PROMPT,
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
                self.logger.info(
                    f"Game number: {req.game_idx}. Team vote phase, Player {req.player_idx} voting on team {team} chosen by {leader}."
                )
            prompt, status = self.agent.vote_on_team(req)
            if status == RequestStatus.TEAM_VOTE_SUCCEED:
                # `rationale` (str): The rationale for the vote.
                # `vote` (bool): The outcome of the vote
                #     (True for "approve", False for "reject").
                resp: Dict[str, str] = req.resp
                resp["prompt"] = prompt
                req.history["team_votes"][-1]["votes"][req.player_idx] = resp
                # need to wait all five
                if not len(req.history["team_votes"][-1]["votes"]) == 5:
                    return
                votes = [
                    vote["vote"]
                    for vote in req.history["team_votes"][-1]["votes"].values()
                ]
                result = req.env.gather_team_votes(votes)
                req.history["team_votes"][-1]["result"] = result[-1]
                approved_votes = sum(votes)
                if self.logger:
                    self.logger.info(
                        f"Game number: {req.game_idx}. {approved_votes} approved, {len(votes) - approved_votes} failed. The team is {'accepted' if result[-1] else 'failed'}."
                    )

                phase: int = req.env.get_phase()[0]
                if phase == 2:
                    for i in range(5):
                        self.process_req(
                            req=Request(
                                prompt=None,
                                resp=None,
                                game_idx=req.game_idx,
                                player_idx=i,
                                history=req.history,
                                env=req.env,
                                status=RequestStatus.QUEST_VOTE_GET_PROMPT,
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
                        f"Game number: {req.game_idx}. {req.player_idx} not in quest team, so this request is ignored. "
                    )
                return

            prompt, status = self.agent.vote_on_mission(req)
            if status == RequestStatus.QUEST_VOTE_SUCCEED:
                # `rationale` (str): The rationale for the vote.
                # `vote` (bool): The outcome of the vote
                #     (True for "approve", False for "reject").
                resp: Dict[str, str] = req.resp
                resp["prompt"] = prompt
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
                    self.logger.info(
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
                        ),
                        req_queue=req_queue,
                    )
                    return
                elif req.env.done:
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
                    )
                )
                return

        elif req.status in [
            RequestStatus.ASSASSIN_GET_PROMPT,
            RequestStatus.ASSASSIN_CHECK_ERROR,
        ]:

            prompt, status = self.agent.assassinate(req)
            if status == RequestStatus.ASSASSIN_VOTE_SUCCEED:
                resp: Dict[str, str] = req.resp
                resp["prompt"] = prompt
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
                    self.logger.info(
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
                    )
                )
                return
        else:
            raise ValueError(f"Unknown status {req.status}")
            # history["team_votes"][0] = {"votes": Dict[player_i -> {"rationale", "vote": T/F, "prompt"}], "result": T/F}


def main(
    model_name,
    output_path,
    n_games=20,
    seed: int = 111,
    max_tokens: int = 512,
    temperature=0,
    top_p=1,
    to_discuss=True,
    add_strategy_in_history=False,
    to_guess_role: bool = False,
    to_guess_multiple_player_role: bool = False,
    to_guess_belief: bool = False,
    use_summary: bool = False,
    n_gpus: int = 1,
):
    seeder = random.Random(seed)

    presets = json.load(open("data/avalon/dev.json"))
    # debug
    # preset = {
    #     "num_players": 5,
    #     "quest_leader": 0,
    #     # "role_names": ["Assassin", "Merlin", "Servant", "Servant", "Minion"],
    #     "role_names": ["Merlin", "Assassin", "Servant", "Servant", "Minion"],
    # }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_name = os.path.split(output_path)[-1].rsplit(".", 1)[0] + ".txt"
    logger = get_logger(
        __name__,
        logger_level="debug",
        console_level="info",
        file_level="debug",
        log_path=os.path.join("logs", log_name),
    )
    if to_guess_role != to_guess_belief:
        raise ValueError(
            "Current only support guess role and guess belief together."
        )

    # init
    reqs = []
    if DEBUG:
        model = VllmWrapper(
            strategy=AnyscaleInferenceStrategy(), model_name=model_name
        )
        sampling_params = SamplingParamsDummy(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
    else:
        model = LLM(
            model=model_name, dtype="float16", tensor_parallel_size=n_gpus
        )
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
    agent = VllmAgent(
        add_strategy_in_history=add_strategy_in_history,
        use_summary=use_summary,
        chat_template=get_conv_template("llama-3"),
    )
    req_processor = RequestProcessor(
        agent=agent,
        to_discuss=to_discuss,
        add_strategy_in_history=False,
        to_guess_role=to_guess_role,
        to_guess_multiple_player_role=to_guess_multiple_player_role,
        n_guess_role_repeat=1,
        to_guess_belief=to_guess_belief,
        use_summary=use_summary,
        logger=logger,
    )
    histories = []
    for game_i in range(1, n_games + 1):
        preset = seeder.choice(presets)
        env = AvalonGameEnvironment.from_presets(preset)
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
            "present": preset,
            "id": game_i,
        }
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
    pbar = tqdm(total=n_games, desc="Sampling games")
    while reqs:
        new_reqs = []
        for req in reqs:
            req_processor.process_req(
                req=req,
                req_queue=new_reqs,
            )
        reqs = new_reqs
        resps = model.generate(
            [req.prompt for req in reqs],
            sampling_params=sampling_params,
            # use_tqdm=False,
        )
        for req, resp in zip(reqs, resps):
            req.resp = resp.outputs[0].text
            if not DEBUG:
                req.history["input_tokens"] += len(resp.prompt_token_ids)
                req.history["output_tokens"] += len(resp.outputs[0].token_ids)
        pbar.n = req_processor.n_finished_games
        pbar.refresh()

        # debug
        with open(output_path, "w") as f:
            f.write(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in histories]
                )
            )
        with open("outputs/reqs.pkl", "wb") as f:
            pickle.dump(reqs, f)
        # debug

    # save all
    with open(output_path, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in histories]
            )
        )
    logger.info(f"{len(histories)} games saved to {output_path}.")


if __name__ == "__main__":
    StrictFire(main)