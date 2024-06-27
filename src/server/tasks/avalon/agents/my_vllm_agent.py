from typing import List, Literal, Dict, Any, Union
from src.utils.misc import (
    format_history,
    get_game_info_prompt,
    parse_json,
    get_player_str,
)
from src.server.tasks.avalon.engine import AvalonBasicConfig
from src.utils.vllm_misc import Request, RequestStatus
from src.server.tasks.avalon.agents.agent import Agent
from src.server.tasks.avalon.my_prompts import (
    INTRODUCTION,
    MERLIN_REVEAL_PROMPT,
    EVIL_REVEAL_PROMPT,
    PERCIVAL_REVEAL_PROMPT,
    SUMMARIZE,
    TEAM_DISCUSSION,
    PROPOSE_TEAM_PROMPT,
    RETRY_JSON_PROMPT,
    PROPOSE_TEAM_INVALID_SIZE_PROMPT,
    PROPOSE_TEAM_INVALID_PLAYER_PROMPT,
    PROPOSE_TEAM_DUPLICATE_PROMPT,
    VOTE_MISSION_ACTION,
    TEAM_VOTE,
    ASSASSINATION_PROMPT,
    GUESS_GOOD_ROLE_PROMPT,
    GUESS_ALL_ROLE_PROMPT,
    GUESS_OTHERS_BELIEF_PRMOPT,
    GUESS_ONE_ROLE_PROMPT,
    GUESS_ROLE_CHEAT_DIFFERENT_HINT,
    GUESS_ROLE_CHEAT_SAME_HINT,
)
from fastchat.conversation import Conversation
import json
import logging
import random

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class VllmAgent:
    def __init__(
        self,
        chat_template: Conversation,
        seed: int = 111,
        add_strategy_in_prompt: bool = False,
        add_quest_strategy_in_prompt: bool = False,
        use_summary: bool = False,
        include_prev_disc: bool = True,
        max_trials: int = 3,
    ):
        """
        Initialize the agent with the given parameters.

        Args:
            add_strategy_in_prompt (bool): Whether to prompt the recommended strategy.
            max_trials (int): The maximum number of trials to restart prompt.
            include_prev_disc (bool): If `use_summary` is false, this param
                controls whether we include previous discussions in the prompt.

        """
        self.chat_template = chat_template
        self.seed = seed
        self.seeder = random.Random(seed)
        self.add_strategy_in_prompt = add_strategy_in_prompt
        self.add_quest_strategy_in_prompt = add_quest_strategy_in_prompt
        self.use_summary = use_summary
        self.include_prev_disc = include_prev_disc
        self.max_trials = max_trials

    def _get_prompt_from_msg(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a prompt from a list of messages.

        Args:
            messages (List[Dict[str, str]]): A list of messages to include in the prompt.

        Returns:
            str: The generated prompt.
        """
        conv = self.chat_template.copy()
        for msg in messages:
            if msg["role"] == "system":
                conv.set_system_message(msg["content"])
            elif msg["role"] == "user":
                conv.append_message(conv.roles[0], msg["content"])
            elif msg["role"] == "assistant":
                conv.append_message(conv.roles[1], msg["content"])
            else:
                raise ValueError(
                    f"{msg['role']} must be one of system, user, assistant."
                )
        assert (
            messages[-1]["role"] == "user"
        ), "Last turn should end with user."
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        del conv.messages[-1]
        return prompt

    def see_sides(self, sides):
        self.player_sides = sides

    def vote_on_team(self, req: Request) -> Dict[str, Union[str, bool]]:
        """
        Vote on a given team.

        Args:
            mission_id (int): The id of the mission.
            team (frozenset[int]): The list of player ids included in the team.

        Returns:
            A dictionary with keys:
                `rationale` (str): The rationale for the vote.
                `vote` (bool): The outcome of the vote
                    (True for "approve", False for "reject").

        Raises:
            OutputException: If the vote outcome is not "approve" or "reject"
                or if the response cannot be parsed as JSON.
        """

        team = req.env.get_current_quest_team()
        if req.status == RequestStatus.TEAM_VOTE_GET_PROMPT:
            prompt = (
                self._get_prompt_prefix(
                    player_id=req.player_idx,
                    history=req.history,
                    player_list=req.env.get_roles(),
                )
                + " "
                + TEAM_VOTE.replace(
                    "{team}", ", ".join([str(p) for p in team])
                )
            )
            req.buffer["msg"] = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            req.buffer["prompt"] = prompt
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.TEAM_VOTE_CHECK_ERROR
        elif req.status == RequestStatus.TEAM_VOTE_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                req.history["n_error"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached for json parsing. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_VOTE_CHECK_ERROR

                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.TEAM_VOTE_CHECK_ERROR
            else:
                for k, t in [("rationale", str), ("vote", str)]:
                    if k not in resp_dict:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.TEAM_VOTE_CHECK_ERROR
                            )
                        err_msg = (
                            f"`{k}` in not included in your JSON response."
                        )
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.TEAM_VOTE_CHECK_ERROR,
                        )
                    elif not isinstance(resp_dict[k], t):
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.TEAM_VOTE_CHECK_ERROR
                            )
                        err_msg = f"`{k}` should be a {t} in your response."
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.TEAM_VOTE_CHECK_ERROR,
                        )

                if resp_dict["vote"] not in ["approve", "reject"]:
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached for team vote error. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_VOTE_CHECK_ERROR
                    error_msg = f"The vote should be either `approve` or `reject`, but you provided `{resp_dict['vote']}`."
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append({"role": "user", "content": error_msg})
                    LOGGER.debug(error_msg + f" Trial: {req.buffer['trial']}")
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_VOTE_CHECK_ERROR
                else:
                    req.resp = resp_dict
                    resp_dict["vote"] = resp_dict["vote"] == "approve"
                    prompt = req.buffer["prompt"]
                    req.resp["prompt"] = prompt
                    req.resp["init_resp"] = req.buffer["init_resp"]
                    return prompt, RequestStatus.TEAM_VOTE_SUCCEED
        else:
            raise ValueError(f"Unknown status: {req.status}")

    def vote_on_mission(self, req: Request) -> bool:
        r"""Vote on a quest (team).

        Args:
            mission_id (int): The id of the mission. num_fails = self.config.num_fails_for_quest[mission_id]
            quest_team (frozenset[int]): The list of player ids included in the quest.

        Returns:
            bool: The vote result.
        """
        quest_team = req.env.get_current_quest_team()
        if req.status == RequestStatus.QUEST_VOTE_GET_PROMPT:
            prompt = (
                self._get_prompt_prefix(
                    player_id=req.player_idx,
                    history=req.history,
                    player_list=req.env.get_roles(),
                )
                + " "
                + VOTE_MISSION_ACTION.replace(
                    "{team}", ", ".join([str(p) for p in quest_team])
                )
            )
            req.buffer["msg"] = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            req.buffer["prompt"] = prompt
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.QUEST_VOTE_CHECK_ERROR
        elif req.status == RequestStatus.QUEST_VOTE_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.history["n_error"] += 1
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached for json parsing. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.QUEST_VOTE_CHECK_ERROR
                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.QUEST_VOTE_CHECK_ERROR
            else:
                for k, t in [("rationale", str), ("vote", str)]:
                    if k not in resp_dict:
                        req.history["n_error"] += 1
                        req.buffer["trial"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.QUEST_VOTE_CHECK_ERROR
                            )
                        err_msg = (
                            f"`{k}` in not included in your JSON response."
                        )
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.QUEST_VOTE_CHECK_ERROR,
                        )
                    elif not isinstance(resp_dict[k], t):
                        req.history["n_error"] += 1
                        req.buffer["trial"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.QUEST_VOTE_CHECK_ERROR
                            )
                        err_msg = f"`{k}` should be a {t} in your response."
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.QUEST_VOTE_CHECK_ERROR,
                        )

                if resp_dict["vote"] not in ["pass", "fail"]:
                    req.history["n_error"] += 1
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached for quest vote outcome. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.QUEST_VOTE_CHECK_ERROR
                    error_msg = f"The vote should be either `pass` or `fail`, but you provided `{resp_dict['vote']}`."
                    LOGGER.debug(
                        f"`{req.resp}` vote should be either `pass` or `fail`. Trial: {req.buffer['trial']}"
                    )
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append({"role": "user", "content": error_msg})
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.QUEST_VOTE_CHECK_ERROR
                else:
                    req.resp = resp_dict
                    resp_dict["vote"] = resp_dict["vote"] == "pass"
                    prompt = req.buffer["prompt"]
                    req.resp["prompt"] = prompt
                    req.resp["init_resp"] = req.buffer["init_resp"]
                    return prompt, RequestStatus.QUEST_VOTE_SUCCEED
        else:
            raise ValueError(f"Unknown status: {req.status}")

    def assassinate(self, req) -> int:
        r"""Assassinate a player.

        Args:
            num_players (int): The number of players in the game.

        Returns:
            int: The id of the player to assassinate. The id is in the range [0, num_players).
        """
        num_players = len(req.env.get_roles())
        if req.status == RequestStatus.ASSASSIN_GET_PROMPT:
            prompt = (
                self._get_prompt_prefix(
                    player_id=req.player_idx,
                    history=req.history,
                    player_list=req.env.get_roles(),
                )
                + "\n\n"
                + ASSASSINATION_PROMPT.replace(
                    "{max_player_id}", str(num_players - 1)
                )
            )
            req.buffer["msg"] = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            req.buffer["prompt"] = prompt
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.ASSASSIN_CHECK_ERROR
        elif req.status == RequestStatus.ASSASSIN_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                req.history["n_error"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached for json parsing. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.ASSASSIN_CHECK_ERROR
                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.ASSASSIN_CHECK_ERROR
            else:
                for k, t in [("merlin", int)]:
                    if k not in resp_dict:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.ASSASSIN_CHECK_ERROR
                            )
                        err_msg = (
                            f"`{k}` in not included in your JSON response."
                        )
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.ASSASSIN_CHECK_ERROR,
                        )
                    elif not isinstance(resp_dict[k], t):
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.ASSASSIN_CHECK_ERROR
                            )
                        err_msg = f"`{k}` should be a {t} in your response."
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.ASSASSIN_CHECK_ERROR,
                        )
                if (
                    resp_dict["merlin"] < 0
                    or resp_dict["merlin"] >= num_players
                ):
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ASSASSIN_CHECK_ERROR
                    err_msg = f"Invalid player id: {resp_dict['merlin']}. Max player id is 4."
                    LOGGER.debug(err_msg + f"Trial: {req.buffer['trial']}")
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append(
                        {
                            "role": "user",
                            # can reuse here
                            "content": PROPOSE_TEAM_INVALID_PLAYER_PROMPT.replace(
                                "{max_player_id}",
                                str(num_players - 1),
                            ),
                        }
                    )
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.ASSASSIN_CHECK_ERROR
                else:
                    req.resp = resp_dict
                    prompt = req.buffer["prompt"]
                    req.resp["prompt"] = prompt
                    req.resp["init_resp"] = req.buffer["init_resp"]
                    return prompt, RequestStatus.ASSASSIN_VOTE_SUCCEED
        else:
            raise ValueError(f"Unknown status: {req.status}")

    def get_believed_sides(self, num_players: int) -> List[float]:
        r"""Get the believed sides of all players.

        Args:
            num_players (int): The number of players in the game.

        Returns:
            List[float]: The list of believed sides (probability) of all players.
        """
        raise NotImplementedError

    def summarize(
        self,
        req: Request,
    ):
        """
        Buffer contains:
            `msg`: if status == 1.
            `prompt`: The orignal prompt to add.
        status (int): _description_
            0: Initial prompting.
            1: Check error.
            200: Success.

        """
        if req.status == RequestStatus.SUMMARIZE_GET_PROMPT:
            prompt = self._get_prompt_prefix(
                player_id=req.player_idx,
                history=req.history,
                player_list=req.env.get_roles(),
            )
            prompt = prompt + " " + SUMMARIZE
            req.buffer["msg"] = [{"role": "user", "content": prompt}]
            req.buffer["prompt"] = prompt
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.SUMMARIZE_CHECK_ERROR
        elif req.status == RequestStatus.SUMMARIZE_CHECK_ERROR:
            if "init_resp" not in req.buffer:
                req.buffer["init_resp"] = req.resp
            prompt = req.buffer["prompt"]
            return prompt, RequestStatus.SUMMARIZE_SUCCEED
        else:
            raise ValueError(f"Unknown status code: {req.status}")

    def _get_prompt_prefix(
        self, history, player_id, player_list, n_rounds_to_skip: int = 0
    ):

        history_str = format_history(
            history,
            player_id if self.add_strategy_in_prompt else None,
            n_rounds_to_skip=n_rounds_to_skip,
            summary_idx=player_id if self.use_summary else None,
            use_summary=self.use_summary,
            include_prev_disc=self.include_prev_disc,
        )
        system_info, identity_prompt, reveal_prompt = get_game_info_prompt(
            player_list=player_list,
            player_id=player_id,
            add_strategy_in_prompt=self.add_strategy_in_prompt,
            add_quest_strategy_in_prompt=self.add_quest_strategy_in_prompt,
        )

        prompt = system_info + "\n\n" + history_str

        prompt += (
            "\n\n### Your Instruction\n"
            + identity_prompt
            + " "
            + reveal_prompt
        )
        return prompt.strip()

    def _retry(self, req, status):
        LOGGER.debug(
            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
        )
        req.buffer["msg"] = req.buffer["msg"][:1]
        prompt = self._get_prompt_from_msg(req.buffer["msg"])
        req.buffer["trial"] = 0
        prompt = self._get_prompt_from_msg(req.buffer["msg"])
        return prompt, status

    def team_discussion(
        self,
        req: Request,
        # team_size, player_id, team_leader_id, history, status: int
    ):
        """
        Buffer contains:
            `msg`: if status == 1.
            `prompt`: The orignal prompt to add.
        status (int): _description_
            0: Initial prompting.
            1: Check error.
            2: Success.

        """
        team_size = req.env.get_team_size()
        team_leader_id: int = req.history["leaders"][-1]
        player_id: int = req.player_idx
        if req.status == RequestStatus.TEAM_DISCUSSION_GET_PROMPT:
            prompt = self._get_prompt_prefix(
                player_id=player_id,
                history=req.history,
                player_list=req.env.get_roles(),
            )
            if player_id == team_leader_id:
                prompt += " You are the Quest leader of this round."
            else:
                prompt += f" Player {team_leader_id} is the Quest leader of this round."
            prompt += f" This Quest requires a team of {team_size} players."

            # history.append(f"Leader is Player {self.history['leaders'][i]}")
            if not any(resp for resp in req.history["team_discs"][-1]):
                prompt += " You are the first to speak in this round."
            prompt += " " + TEAM_DISCUSSION
            req.buffer["msg"] = [{"role": "user", "content": prompt}]
            req.buffer["prompt"] = prompt
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
        elif req.status == RequestStatus.TEAM_DISCUSSION_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.history["n_error"] += 1
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    return self._retry(
                        req, RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
                    )
                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
            else:
                for k in ["strategy", "response"]:
                    if k not in resp:
                        req.history["n_error"] += 1
                        req.buffer["trial"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
                            )
                        err_msg = (
                            f"`{k}` in not included in your JSON response."
                        )
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.TEAM_DISCUSSION_CHECK_ERROR,
                        )
                    elif not isinstance(resp[k], str):
                        req.history["n_error"] += 1
                        req.buffer["trial"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
                            )
                        err_msg = f"`{k}` should be a string in your response."
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.TEAM_DISCUSSION_CHECK_ERROR,
                        )

                req.resp = resp
                prompt = req.buffer["prompt"]
                req.resp["prompt"] = prompt
                req.resp["init_resp"] = req.buffer["init_resp"]
                req.buffer = {}
                return prompt, RequestStatus.TEAM_DISCUSSION_SUCCEED
        else:
            raise ValueError(f"Unknown status: {req.status}")

    def propose_team(
        self,
        req,
    ) -> Dict[str, Union[str, List[int]]]:
        team_size = req.env.get_team_size()
        n_players = len(req.env.get_roles())
        if req.status == RequestStatus.TEAM_PROPOSAL_GET_PROMPT:
            prompt = self._get_prompt_prefix(
                player_id=req.player_idx,
                history=req.history,
                player_list=req.env.get_roles(),
            )
            prompt += " " + PROPOSE_TEAM_PROMPT.replace(
                "{num_player}", str(team_size)
            ).replace("{max_player_id}", f"{n_players-1}")

            req.buffer["msg"] = [{"role": "user", "content": prompt}]
            req.buffer["prompt"] = prompt
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
        elif req.status == RequestStatus.TEAM_PROPOSAL_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                req.history["n_error"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
            else:
                for k, t in [("rationale", str), ("team", list)]:
                    if k not in resp_dict:
                        req.history["n_error"] += 1
                        req.buffer["trial"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                            )
                        err_msg = (
                            f"`{k}` in not included in your JSON response."
                        )
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.TEAM_PROPOSAL_CHECK_ERROR,
                        )
                    elif not isinstance(resp_dict[k], t):
                        req.history["n_error"] += 1
                        req.buffer["trial"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                            )
                        err_msg = f"`{k}` should be a {t} in your response."
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.TEAM_PROPOSAL_CHECK_ERROR,
                        )
                if len(resp_dict["team"]) != team_size:
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    err_msg = f"Team size not matched. We need a team with {team_size} players, but received {resp_dict['team']}."
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append(
                        {
                            "role": "user",
                            "content": PROPOSE_TEAM_INVALID_SIZE_PROMPT.replace(
                                "{target_num_player}", str(team_size)
                            ).replace(
                                "{num_player}", str(len(resp_dict["team"]))
                            ),
                        }
                    )
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                elif len(set(resp_dict["team"])) != team_size:
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    err_msg = f"Duplicate members found on the team. We need a team with {team_size} unique players, but received {resp_dict['team']}."
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append(
                        {
                            "role": "user",
                            "content": PROPOSE_TEAM_DUPLICATE_PROMPT.replace(
                                "{target_num_player}", str(team_size)
                            ),
                        }
                    )
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                elif any(
                    not isinstance(mem, int) or mem < 0 or mem >= n_players
                    for mem in resp_dict["team"]
                ):
                    err_msg = f"Proposed team contains invalid player ids: {resp_dict['team']}. Max player id is 4."
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    n_players = len(req.env.get_roles())
                    messages.append(
                        {
                            "role": "user",
                            "content": PROPOSE_TEAM_INVALID_PLAYER_PROMPT.replace(
                                "{max_player_id}",
                                f"{n_players-1}",
                            ),
                        }
                    )
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                elif any(k not in resp_dict for k in ["rationale", "team"]):
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    keys = []
                    if "rationale" not in resp_dict:
                        keys.append("rationale")
                    if "team" not in resp_dict:
                        keys.append("team")
                    err_msg = f"{keys} should be included in your answer."
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append(
                        {
                            "role": "user",
                            "content": err_msg,
                        }
                    )
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                else:
                    req.resp = resp_dict
                    prompt = req.buffer["prompt"]
                    req.resp["prompt"] = prompt
                    req.resp["init_resp"] = req.buffer["init_resp"]
                    return prompt, RequestStatus.TEAM_PROPOSAL_SUCCEED
        else:
            raise ValueError(f"Unknown status: {req.status}")

    def guess_role_dpo_wo_env(
        self,
        req: Request,
        to_guess_multiple_player: bool = True,
    ) -> Dict[str, Any]:
        """
        Guess the role of a player based on the game state.

        Args:
            player_i (int): The index of the player to guess.


        Returns:
            A dictionary with keys:
                - "output": A dictionary containing the guessed role
                    information.
                - "prompt": The prompt used to generate the response.

        Raises:
            ValueError: If the current role is not one of the allowed roles to guess.
            OutputException: If the response cannot be parsed as JSON after multiple trials.

        """
        roles = req.history["roles"]
        role = roles[req.player_idx]
        # tgt_role = req.tgt_role
        # tgt_player_i = req.tgt_player_i
        good_roles: List[str] = [role[1] for role in roles if role[2]]
        if req.status == RequestStatus.ROLE_GUESS_GET_PROMPT:
            if role[1] == "Merlin":
                raise ValueError("Merlin should not guess other's role.")
            elif role[2]:
                # randomly pick another player
                player_i = self.seeder.choice(
                    [i for i in range(len(roles)) if i != req.player_idx]
                )
                tgt_role: str = req.history["roles"][player_i][1]
                # sample a false role
                if self.seeder.random() < 0.5:
                    tgt_roles = set([role[1] for role in roles]) - set(
                        [tgt_role]
                    )
                    tgt_role = self.seeder.choice(list(tgt_roles))
            elif not role[2]:
                player_i, tgt_role = self.seeder.choice(
                    [
                        (i, role[1])
                        for i, role in enumerate(req.history["roles"])
                        if role[2]
                    ]
                )
                # sample a false role
                if self.seeder.random() < 0.5:
                    tgt_roles = set(good_roles) - set([tgt_role])
                    tgt_role = self.seeder.choice(list(tgt_roles))

            prompt = self._get_prompt_prefix(
                player_id=req.player_idx,
                history=req.history,
                player_list=roles,
            )
            included_roles = []
            if not to_guess_multiple_player:
                prompt += " " + GUESS_ONE_ROLE_PROMPT.replace(
                    "{i}", str(player_i)
                ).replace("{role}", tgt_role)
            else:
                if role[1] == "Servant":
                    prompt += " " + GUESS_ALL_ROLE_PROMPT.replace(
                        "{i}", str(player_i)
                    )
                    included_roles = ["Merlin", "Servant", "Minion"]
                elif role[1] in [
                    "Assassin",
                    "Minion",
                ]:
                    prompt += " " + GUESS_GOOD_ROLE_PROMPT.replace(
                        "{i}", str(player_i)
                    )
                    included_roles = ["Merlin", "Servant"]
                else:
                    raise ValueError(
                        "Merlin can't guess role since he already know."
                    )

            messages = [{"role": "user", "content": prompt}]
            s = "First state your rationale and then provide the score."
            prefix, suffix = prompt.split(s)
            gold_role = roles[player_i]
            if gold_role[1] == tgt_role:
                dpo_prompt = (
                    prefix
                    + GUESS_ROLE_CHEAT_SAME_HINT.replace(
                        "{player_idx}", str(player_i)
                    ).replace("{right_role}", gold_role[1])
                    + " "
                    + s
                    + suffix
                )
            else:
                dpo_prompt = (
                    prefix
                    + GUESS_ROLE_CHEAT_DIFFERENT_HINT.replace(
                        "{player_idx}", str(player_i)
                    )
                    .replace("{right_role}", gold_role[1])
                    .replace("{wrong_role}", tgt_role)
                    + " "
                    + s
                    + suffix
                )
            dpo_messages = [{"role": "user", "content": dpo_prompt}]

            req.buffer["tgt_role"] = tgt_role
            req.buffer["tgt_player_i"] = player_i
            req.buffer["included_roles"] = included_roles
            req.buffer["prompt"] = [prompt, dpo_prompt]
            req.buffer["msg"] = [messages, dpo_messages]
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"][0])
            dpo_prompt = self._get_prompt_from_msg(req.buffer["msg"][1])
            # need to self req.buffer['is_dpo'] = True/False
            return [prompt, dpo_prompt], RequestStatus.ROLE_GUESS_CHECK_ERROR
        elif req.status == RequestStatus.ROLE_GUESS_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                req.history["n_error"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    if req.buffer["is_dpo"]:
                        req.buffer["msg"][1] = req.buffer["msg"][1][:1]
                        prompt = self._get_prompt_from_msg(
                            req.buffer["msg"][1]
                        )
                    else:
                        req.buffer["msg"][0] = req.buffer["msg"][0][:1]
                        prompt = self._get_prompt_from_msg(
                            req.buffer["msg"][0]
                        )
                    req.buffer["trial"] = 0
                    return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                if req.buffer["is_dpo"]:
                    messages = req.buffer["msg"][1]
                else:
                    messages = req.buffer["msg"][0]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                if req.buffer["is_dpo"]:
                    req.buffer["msg"][1] = messages
                else:
                    req.buffer["msg"][0] = messages
                prompt = self._get_prompt_from_msg(messages)
                return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
            else:
                if "score" not in resp_dict:
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        if req.buffer["is_dpo"]:
                            messages = req.buffer["msg"][1][:1]
                        else:
                            messages = req.buffer["msg"][0][:1]
                        req.buffer["msg"] = messages
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(messages)
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                    err_msg = "Your response should follow the specified JSON format. It doesn't contain the key `score`."
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    if req.buffer["is_dpo"]:
                        messages = req.buffer["msg"][1]
                    else:
                        messages = req.buffer["msg"][0]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": req.resp,
                        }
                    )
                    messages.append({"role": "user", "content": err_msg})
                    req.buffer["msg"] = messages
                    if req.buffer["is_dpo"]:
                        req.buffer["msg"][0] = messages
                    else:
                        req.buffer["msg"][1] = messages
                    prompt = self._get_prompt_from_msg(messages)
                    return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                elif "rationale" not in resp_dict:
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]

                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                    err_msg = "Your response should follow the specified JSON format. It doesn't contain the key `rationale`."
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    if req.buffer["is_dpo"]:
                        messages = req.buffer["msg"][1]
                    else:
                        messages = req.buffer["msg"][0]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": req.resp,
                        }
                    )
                    messages.append({"role": "user", "content": err_msg})
                    if req.buffer["is_dpo"]:
                        req.buffer["msg"][1] = messages
                    else:
                        req.buffer["msg"][0] = messages
                    prompt = self._get_prompt_from_msg(messages)
                    return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                elif (
                    not isinstance(resp_dict["score"], int)
                    or resp_dict["score"] < 1
                    or resp_dict["score"] > 10
                ):
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        if req.buffer["is_dpo"]:
                            messages = req.buffer["msg"][1][:1]
                            req.buffer["msg"][1] = messages
                        else:
                            messages = req.buffer["msg"][0][:1]
                            req.buffer["msg"][0] = messages
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(messages)
                        return (
                            prompt,
                            RequestStatus.ROLE_GUESS_CHECK_ERROR,
                        )
                    err_msg = "Your response should provide an integer score from 1 to 10."
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    if req.buffer["is_dpo"]:
                        messages = req.buffer["msg"][1]
                    else:
                        messages = req.buffer["msg"][0]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": req.resp,
                        }
                    )
                    messages.append({"role": "user", "content": err_msg})
                    if req.buffer["is_dpo"]:
                        req.buffer["msg"][1] = messages
                    else:
                        req.buffer["msg"][0] = messages
                    prompt = self._get_prompt_from_msg(messages)
                    return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                req.resp = resp_dict
                if req.buffer["is_dpo"]:
                    prompt = req.buffer["prompt"][1]
                else:
                    prompt = req.buffer["prompt"][0]
                return prompt, RequestStatus.ROLE_GUESS_SUCCEED

    def guess_role_wo_env(
        self,
        req: Request,
        to_guess_multiple_player: bool = True,
    ) -> Dict[str, Any]:
        """
        Guess the role of a player based on the game state.

        Args:
            player_i (int): The index of the player to guess.


        Returns:
            A dictionary with keys:
                - "output": A dictionary containing the guessed role
                    information.
                - "prompt": The prompt used to generate the response.

        Raises:
            ValueError: If the current role is not one of the allowed roles to guess.
            OutputException: If the response cannot be parsed as JSON after multiple trials.

        """
        roles = req.history["roles"]
        role = roles[req.player_idx]
        tgt_role = req.tgt_role
        tgt_player_i = req.tgt_player_i
        # good_roles: List[str] = [
        #     role[1] for role in req.env.get_roles() if role[2]
        # ]
        # bad_roles: List[str] = [
        #     role[1] for role in req.env.get_roles() if not role[2]
        # ]
        if req.status == RequestStatus.ROLE_GUESS_GET_PROMPT:
            if role[1] == "Merlin":
                raise ValueError("Merlin should not guess other's role.")

            prompt = self._get_prompt_prefix(
                player_id=req.player_idx,
                history=req.history,
                player_list=roles,
            )
            included_roles = []
            if not to_guess_multiple_player:
                prompt += " " + GUESS_ONE_ROLE_PROMPT.replace(
                    "{i}", str(tgt_player_i)
                ).replace("{role}", tgt_role)
            else:
                if role[1] == "Servant":
                    prompt += " " + GUESS_ALL_ROLE_PROMPT.replace(
                        "{i}", str(tgt_player_i)
                    )
                    included_roles = ["Merlin", "Servant", "Minion"]
                elif role[1] in [
                    "Assassin",
                    "Minion",
                ]:
                    prompt += " " + GUESS_GOOD_ROLE_PROMPT.replace(
                        "{i}", str(tgt_player_i)
                    )
                    included_roles = ["Merlin", "Servant"]
                else:
                    raise ValueError(
                        "Merlin can't guess role since he already know."
                    )
            messages = [{"role": "user", "content": prompt}]
            req.buffer["tgt_role"] = tgt_role
            req.buffer["tgt_player_i"] = tgt_player_i
            req.buffer["included_roles"] = included_roles
            req.buffer["prompt"] = prompt
            req.buffer["msg"] = messages
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
        elif req.status == RequestStatus.ROLE_GUESS_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                req.history["n_error"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
            else:
                if to_guess_multiple_player:
                    role_error = [
                        role
                        for role in req.buffer["included_roles"]
                        if role not in resp_dict
                    ]
                    if role_error:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                        err_msg = f"Your response should follow the specified JSON format. It doesn't contain the key `{role_error[0]}`. Received: {json.dumps(resp_dict, indent=4)}"
                        LOGGER.debug(
                            err_msg + f" Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                else:
                    if "score" not in resp_dict:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                        err_msg = "Your response should follow the specified JSON format. It doesn't contain the key `score`."
                        LOGGER.debug(
                            err_msg + f" Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                    elif "rationale" not in resp_dict:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                        err_msg = "Your response should follow the specified JSON format. It doesn't contain the key `rationale`."
                        LOGGER.debug(
                            err_msg + f" Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                    elif (
                        not isinstance(resp_dict["score"], int)
                        or resp_dict["score"] < 1
                        or resp_dict["score"] > 10
                    ):
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return (
                                prompt,
                                RequestStatus.ROLE_GUESS_CHECK_ERROR,
                            )
                        err_msg = "Your response should provide an integer score from 1 to 10."
                        LOGGER.debug(
                            err_msg
                            + f" Received: {req.resp}. Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                req.resp = resp_dict
                prompt = req.buffer["prompt"]
                return prompt, RequestStatus.ROLE_GUESS_SUCCEED

    def guess_role(
        self,
        req: Request,
        to_guess_multiple_player: bool = True,
    ) -> Dict[str, Any]:
        """
        Guess the role of a player based on the game state.

        Args:
            player_i (int): The index of the player to guess.


        Returns:
            A dictionary with keys:
                - "output": A dictionary containing the guessed role
                    information.
                - "prompt": The prompt used to generate the response.

        Raises:
            ValueError: If the current role is not one of the allowed roles to guess.
            OutputException: If the response cannot be parsed as JSON after multiple trials.

        """
        n_rounds_to_skip = 0
        n_players = len(req.env.get_roles())
        role_name: str = req.history["roles"][req.player_idx][1]
        good_roles: List[str] = [
            role[1] for role in req.env.get_roles() if role[2]
        ]
        bad_roles: List[str] = [
            role[1] for role in req.env.get_roles() if not role[2]
        ]
        if req.status == RequestStatus.ROLE_GUESS_GET_PROMPT:
            if "tgt_player_i" in req.args:
                player_i = req.args["tgt_player_i"]
                tgt_role: str = req.history["roles"][player_i][1]
            elif role_name == "Merlin":
                raise ValueError("Merlin should not guess other's role.")
            elif role_name in good_roles:
                # randomly pick another player
                player_i = self.seeder.choice(
                    [i for i in range(n_players) if i != req.player_idx]
                )
                tgt_role: str = req.history["roles"][player_i][1]
                # sample a false role
                if self.seeder.random() < 0.5:
                    tgt_roles = set(
                        [role[1] for role in req.env.get_roles()]
                    ) - set([tgt_role])
                    tgt_role = self.seeder.choice(list(tgt_roles))
            elif role_name in bad_roles:
                player_i, tgt_role = self.seeder.choice(
                    [
                        (i, role[1])
                        for i, role in enumerate(req.history["roles"])
                        if role[1] in good_roles
                    ]
                )
                # sample a false role
                if self.seeder.random() < 0.5:
                    tgt_roles = set(good_roles) - set([tgt_role])
                    tgt_role = self.seeder.choice(list(tgt_roles))

            prompt = self._get_prompt_prefix(
                player_id=req.player_idx,
                history=req.history,
                player_list=req.env.get_roles(),
            )
            included_roles = []
            if not to_guess_multiple_player:
                prompt += " " + GUESS_ONE_ROLE_PROMPT.replace(
                    "{i}", str(player_i)
                ).replace("{role}", tgt_role)
            else:
                if role_name == "Servant":
                    prompt += " " + GUESS_ALL_ROLE_PROMPT.replace(
                        "{i}", str(player_i)
                    )
                    included_roles = ["Merlin", "Servant", "Minion"]
                elif role_name in [
                    "Assassin",
                    "Minion",
                ]:
                    prompt += " " + GUESS_GOOD_ROLE_PROMPT.replace(
                        "{i}", str(player_i)
                    )
                    included_roles = ["Merlin", "Servant"]
                else:
                    raise ValueError(
                        "Merlin can't guess role since he already know."
                    )
            messages = [{"role": "user", "content": prompt}]
            req.buffer["tgt_role"] = tgt_role
            req.buffer["tgt_player_i"] = player_i
            req.buffer["included_roles"] = included_roles
            req.buffer["prompt"] = prompt
            req.buffer["msg"] = messages
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
        elif req.status == RequestStatus.ROLE_GUESS_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                req.history["n_error"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                LOGGER.debug(
                    f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                )
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
            else:
                if to_guess_multiple_player:
                    role_error = [
                        role
                        for role in req.buffer["included_roles"]
                        if role not in resp_dict
                    ]
                    if role_error:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            req.buffer["prompt"] = prompt
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                        err_msg = f"Your response should follow the specified JSON format. It doesn't contain the key `{role_error[0]}`."
                        LOGGER.debug(
                            err_msg + f" Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                else:
                    if "score" not in resp_dict:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            req.buffer["prompt"] = prompt
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                        err_msg = "Your response should follow the specified JSON format. It doesn't contain the key `score`."
                        LOGGER.debug(
                            err_msg + f" Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                    elif "rationale" not in resp_dict:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            req.buffer["prompt"] = prompt
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                        err_msg = "Your response should follow the specified JSON format. It doesn't contain the key `rationale`."
                        LOGGER.debug(
                            err_msg + f" Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                    elif (
                        not isinstance(resp_dict["score"], int)
                        or resp_dict["score"] < 1
                        or resp_dict["score"] > 10
                    ):
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            LOGGER.debug(
                                f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                            )
                            req.buffer["msg"] = req.buffer["msg"][:1]
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            req.buffer["prompt"] = prompt
                            req.buffer["trial"] = 0
                            prompt = self._get_prompt_from_msg(
                                req.buffer["msg"]
                            )
                            return (
                                prompt,
                                RequestStatus.ROLE_GUESS_CHECK_ERROR,
                            )
                        err_msg = "Your response should provide an integer score from 1 to 10."
                        LOGGER.debug(
                            err_msg + f" Trial: {req.buffer['trial']}"
                        )
                        messages = req.buffer["msg"]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": req.resp,
                            }
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_GUESS_CHECK_ERROR
                req.resp = resp_dict
                prompt = req.buffer["prompt"]
                return prompt, RequestStatus.ROLE_GUESS_SUCCEED

    def guess_belief(self, req: Request) -> Dict[str, Any]:
        n_players = len(req.env.get_roles())
        if "tgt_player_i" in req.args:
            tgt_player_i = req.args["tgt_player_i"]
        else:
            tgt_player_i = self.seeder.choice(
                [i for i in range(n_players) if i != req.player_idx]
            )
        all_roles = list(set([role[1] for role in req.env.get_roles()]))
        tgt_role = self.seeder.choice(all_roles)
        if req.status == RequestStatus.ROLE_BELIEF_GET_PROMPT:
            prompt = self._get_prompt_prefix(
                player_id=req.player_idx,
                history=req.history,
                player_list=req.env.get_roles(),
            )
            prompt += " " + GUESS_OTHERS_BELIEF_PRMOPT.replace(
                "{i}", str(tgt_player_i)
            ).replace("{role}", tgt_role)
            messages = [{"role": "user", "content": prompt}]
            req.buffer["trial"] = 0
            req.buffer["prompt"] = prompt
            req.buffer["tgt_player_i"] = tgt_player_i
            req.buffer["tgt_role"] = tgt_role
            req.buffer["msg"] = messages
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.ROLE_BELIEF_CHECK_ERROR
        elif req.status == RequestStatus.ROLE_BELIEF_CHECK_ERROR:
            try:
                if "init_resp" not in req.buffer:
                    req.buffer["init_resp"] = req.resp
                resp_dict: Dict[str, str] = parse_json(req.resp)
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                req.history["n_error"] += 1

                if req.buffer["trial"] >= self.max_trials:
                    self._retry(req, RequestStatus.ROLE_BELIEF_CHECK_ERROR)
                err_msg = f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                LOGGER.debug(err_msg)
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.ROLE_BELIEF_CHECK_ERROR
            else:
                for k, t in [("rationale", str), ("score", int)]:
                    if k not in resp_dict:
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.ROLE_BELIEF_CHECK_ERROR
                            )
                        err_msg = (
                            f"`{k}` in not included in your JSON response."
                        )
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.ROLE_BELIEF_CHECK_ERROR,
                        )
                    elif not isinstance(resp_dict[k], t):
                        req.buffer["trial"] += 1
                        req.history["n_error"] += 1
                        if req.buffer["trial"] >= self.max_trials:
                            return self._retry(
                                req, RequestStatus.ROLE_BELIEF_CHECK_ERROR
                            )
                        err_msg = f"`{k}` should be a {t} in your response."
                        LOGGER.debug(f"{err_msg} Trial: {req.buffer['trial']}")
                        messages = req.buffer["msg"]
                        messages.append(
                            {"role": "assistant", "content": req.resp}
                        )
                        messages.append({"role": "user", "content": err_msg})
                        req.buffer["msg"] = messages
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return (
                            prompt,
                            RequestStatus.ROLE_BELIEF_CHECK_ERROR,
                        )

                if resp_dict["score"] < 1 or resp_dict["score"] > 10:
                    req.buffer["trial"] += 1
                    req.history["n_error"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        return self._retry(
                            req, RequestStatus.ROLE_BELIEF_CHECK_ERROR
                        )
                    messages = req.buffer["msg"]
                    err_msg = f"score must be from 1 to 10. Received: {resp_dict['score']}."
                    LOGGER.debug(err_msg + f" Trial: {req.buffer['trial']}")
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append(
                        {
                            "role": "user",
                            "content": err_msg,
                        }
                    )
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.ROLE_BELIEF_CHECK_ERROR
            req.resp = resp_dict
            prompt = req.buffer["prompt"]
            return prompt, RequestStatus.ROLE_BELIEF_SUCCEED
