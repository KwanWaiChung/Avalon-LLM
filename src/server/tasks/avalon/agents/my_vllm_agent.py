from typing import List, Literal, Dict, Any, Union
from src.server.tasks.avalon.engine import AvalonBasicConfig
from src.utils.vllm_misc import Request, RequestStatus
from src.server.tasks.avalon.agents.agent import Agent
from src.server.tasks.avalon.my_prompts import (
    INTRODUCTION,
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
        add_strategy_in_history: bool = False,
        use_summary: bool = False,
        max_trials: int = 3,
    ):
        """
        Initialize the agent with the given parameters.

        Args:
            to_recommend_strategy (bool): Whether to prompt the recommended strategy.
            max_trials (int): The maximum number of trials to restart prompt.


        """
        self.chat_template = chat_template
        self.seed = seed
        self.seeder = random.Random(111)
        self.add_strategy_in_history = add_strategy_in_history
        self.use_summary = use_summary
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

    def get_game_info_prompt(self, player_list, player_id) -> None:
        """Initiliaze the game info for the agent, which includes game introduction, role, and reveal information for different roles."""
        # Introduction Prompt
        verbal_side = ["Evil", "Good"]
        intro_prompt = INTRODUCTION
        intro_prompt += "\n"

        minion = ""
        servant_list = []
        assassin = ""
        merlin = ""
        name = f"Player {player_id}"
        role_name = player_list[player_id][1]
        for idx, player_info in enumerate(player_list):
            if player_info[1] == "Minion":
                minion = str(idx)
            elif player_info[1] == "Servant":
                servant_list.append(str(idx))
            elif player_info[1] == "Assassin":
                assassin = str(idx)
            elif player_info[1] == "Merlin":
                merlin = str(idx)
            else:
                raise ValueError(f"Unrecognized role: {player_info[1]}")

        identity_prompt = f"You are {name}, with identity {role_name}."

        reveal_prompt = ""
        good_team = sorted(servant_list + [merlin])
        if role_name == "Merlin":
            reveal_prompt = (
                f"You know that Players {minion} and {assassin} are Evil, and Players"
                f" {', '.join(servant_list)} are Servants."
            )
        elif role_name == "Minion":
            reveal_prompt = (
                f"You know that Player {assassin} is Assassin, and Players"
                f" {', '.join(good_team)} are on the good team, but you do not know who is Merlin."
            )
        elif role_name == "Assassin":
            reveal_prompt = (
                f"You know that Player {minion} is Minion, and Players"
                f" {', '.join(good_team)} are on the good team, but you do not know who is Merlin."
            )

        system_info = intro_prompt.strip()
        return system_info, identity_prompt, reveal_prompt

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
                resp_dict: Dict[str, str] = json.loads(
                    "{"
                    + req.resp.split("```json")[-1]
                    .split("```")[0]
                    .split("{", 1)[-1]
                )
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached for json parsing. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    req.buffer["prompt"] = prompt
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
                if resp_dict["vote"] not in ["approve", "reject"]:
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached for team vote error. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_VOTE_CHECK_ERROR
                    error_msg = f"The vote should be either `approve` or `reject`, but you provided `{resp_dict['vote']}`."
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append({"role": "user", "content": error_msg})
                    LOGGER.debug(error_msg)
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_VOTE_CHECK_ERROR
                else:
                    req.resp = resp_dict
                    resp_dict["vote"] = resp_dict["vote"] == "approve"
                    prompt = req.buffer["prompt"]
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
                resp_dict: Dict[str, str] = json.loads(
                    "{"
                    + req.resp.split("```json")[-1]
                    .split("```")[0]
                    .split("{", 1)[-1]
                )
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached for json parsing. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    req.buffer["prompt"] = prompt
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
                return prompt, RequestStatus.QUEST_VOTE_CHECK_ERROR
            else:
                if resp_dict["vote"] not in ["pass", "fail"]:
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached for quest vote outcome. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.QUEST_VOTE_CHECK_ERROR
                    error_msg = f"The vote should be either `pass` or `fail`, but you provided `{resp_dict['vote']}`."
                    LOGGER.debug(
                        f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
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
        num_players = 5
        if req.status == RequestStatus.ASSASSIN_GET_PROMPT:
            prompt = (
                self._get_prompt_prefix(
                    player_id=req.player_idx,
                    history=req.history,
                    player_list=req.env.get_roles(),
                )
                + "\n\n"
                + ASSASSINATION_PROMPT.replace(
                    "{max_player_id}", str(num_players)
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
                resp_dict: Dict[str, str] = json.loads(
                    "{"
                    + req.resp.split("```json")[-1]
                    .split("```")[0]
                    .split("{", 1)[-1]
                )
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached for json parsing. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    req.buffer["prompt"] = prompt
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
                if (
                    resp_dict["merlin"] < 0
                    or resp_dict["merlin"] >= num_players
                ):
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ASSASSIN_CHECK_ERROR
                    err_msg = f"Invalid player id: {resp_dict['merlin']}. Max player id is 4."
                    LOGGER.debug(err_msg)
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
            prompt = req.buffer["prompt"]
            return prompt, RequestStatus.SUMMARIZE_SUCCEED
        else:
            raise ValueError(f"Unknown status code: {req.status}")

    @staticmethod
    def get_history_str(
        history: Dict[str, Any],
        strategy_idx: int = None,
        n_rounds_to_skip: int = 0,
        summary_idx: int = None,
        use_summary: bool = False,
    ) -> str:
        output = ["### Game Play History"]
        n_round = len(history["leaders"])
        start_round = 0

        if (
            use_summary
            and history["summaries"]
            and summary_idx in history["summaries"][0]
        ):
            # This last condition is just  to ensure we have at least
            # one summary. We either use the summary in the last round
            # if it exists, or second last round when summarizing, because
            # an empty place holder is inserted.
            # history['summaries'][0][0] = {'prompt': ..., 'resp': ...}
            if summary_idx in history["summaries"][-1]:
                start_round = len(history["summaries"]) - 1
            else:
                assert summary_idx in history["summaries"][-2]
                start_round = len(history["summaries"]) - 2
            output.append("\n#### Previous Game Play Summary")
            output.append(
                history["summaries"][start_round][summary_idx]["resp"]
            )

        for i in range(start_round, n_round):
            if i < n_rounds_to_skip:
                continue
            # history.append(f"Leader is Player {history['leaders'][i]}")
            include_cur_round_diss = (
                use_summary
                and (
                    i >= len(history["summaries"])
                    or summary_idx not in history["summaries"][i]
                )
            ) or not use_summary
            if include_cur_round_diss and any(
                resp for resp in history["team_discs"][i].values()
            ):
                output.append(f"\n#### Round {i + 1} Discussion")
                if strategy_idx is not None and strategy_idx < len(
                    history["team_discs"][i]
                ):
                    output.append(
                        f"**Strategy:** {history['team_discs'][i][strategy_idx]['strategy']}"
                    )
                for p_i, resp in enumerate(history["team_discs"][i].values()):
                    if resp:
                        output.append(f"Player {p_i}: {resp['response']}")

            if i < len(history["team_props"]):
                output.append(f"\n#### Round {i+1} Proposed Team")
                if (
                    strategy_idx is not None
                    and history["leaders"][i] == strategy_idx
                ):
                    output.append(
                        f"**Strategy:** {history['team_props'][i]['rationale']}"
                    )
                players = []
                for player in history["team_props"][i]["team"]:
                    players.append(f"Player {player}")
                output.append(
                    f"The leader, Player {history['leaders'][i]}, proposed "
                    + ", ".join(players[:-1])
                    + ", and "
                    + players[-1]
                    + "."
                )

            if i < len(history["team_votes"]):
                output.append(f"\n#### Round {i+1} Team Votes")
                if strategy_idx is not None:
                    output.append(
                        f"**Strategy:** {history['team_votes'][i]['votes'][strategy_idx]['rationale']}"
                    )
                    output.append(
                        f"**Your Vote:** {'Approve' if history['team_votes'][i]['votes'][strategy_idx]['vote'] else 'reject'}."
                    )
                num_approves = sum(
                    vote["vote"]
                    for vote in history["team_votes"][i]["votes"].values()
                )

                output.append(
                    f"{num_approves} player(s)"
                    " approved,"
                    f" {len(history['team_votes'][i]['votes']) - num_approves} player(s)"
                    " rejected."
                )
                output.append(
                    "Team result: The proposed team is"
                    f" {'approved' if history['team_votes'][i]['result'] else 'rejected'}."
                )

            if (
                i < len(history["team_votes"])
                and history["team_votes"][i]["result"]
                and i < len(history["quest_votes"])
                and history["quest_votes"][i]
            ):
                output.append(f"\n#### Round {i+1} Quest Votes")
                if (
                    strategy_idx is not None
                    and strategy_idx in history["team_props"][i]["team"]
                ):
                    _idx = history["team_props"][i]["team"].index(strategy_idx)
                    output.append(
                        f"**Strategy:** {history['quest_votes'][i]['votes'][_idx]['rationale']}"
                    )

                num_approves = sum(
                    vote["vote"]
                    for vote in history["quest_votes"][i]["votes"].values()
                )
                output.append(
                    f"{num_approves} player(s)"
                    " passed,"
                    f" {len(history['quest_votes'][i]['votes']) - num_approves} player(s)"
                    " failed."
                )
                output.append(
                    "Quest result: The mission"
                    f" {'succeeded' if history['quest_votes'][i]['result'] else 'failed'}."
                )
        history_str = "\n".join(output)
        if len(output) == 1:
            history_str += "\nNone."
        return history_str.strip()

    def _get_prompt_prefix(
        self, history, player_id, player_list, n_rounds_to_skip: int = 0
    ):

        history_str = self.get_history_str(
            history,
            player_id if self.add_strategy_in_history else None,
            n_rounds_to_skip=n_rounds_to_skip,
            use_summary=self.use_summary,
            summary_idx=player_id if self.use_summary else None,
        )
        system_info, identity_prompt, reveal_prompt = (
            self.get_game_info_prompt(
                player_list=player_list, player_id=player_id
            )
        )
        prompt = system_info + "\n\n" + history_str

        prompt += (
            "\n\n### Your Instruction\n"
            + identity_prompt
            + " "
            + reveal_prompt
        )
        return prompt.strip()

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
        team_size = 5
        n_round = len(req.history["leaders"])
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
            prompt += f" This Quest requires {team_size} players to vote."

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
                resp: Dict[str, str] = json.loads(
                    "{"
                    + req.resp.split("```json")[-1]
                    .split("```")[0]
                    .split("{", 1)[-1]
                )
                req.resp = resp
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    req.buffer["prompt"] = prompt
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
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
                prompt = req.buffer["prompt"]
                req.buffer = {}
                return prompt, RequestStatus.TEAM_DISCUSSION_SUCCEED
        else:
            raise ValueError(f"Unknown status: {req.status}")

    def propose_team(
        self,
        req,
    ) -> Dict[str, Union[str, List[int]]]:
        team_size = req.env.get_team_size()
        if req.status == RequestStatus.TEAM_PROPOSAL_GET_PROMPT:
            prompt = self._get_prompt_prefix(
                player_id=req.player_idx,
                history=req.history,
                player_list=req.env.get_roles(),
            )
            prompt += " " + PROPOSE_TEAM_PROMPT.replace(
                "{num_player}", str(team_size)
            ).replace("{max_player_id}", "4")

            req.buffer["msg"] = [{"role": "user", "content": prompt}]
            req.buffer["prompt"] = prompt
            req.buffer["trial"] = 0
            prompt = self._get_prompt_from_msg(req.buffer["msg"])
            return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
        elif req.status == RequestStatus.TEAM_PROPOSAL_CHECK_ERROR:
            try:
                resp_dict: Dict[str, str] = json.loads(
                    "{"
                    + req.resp.split("```json")[-1]
                    .split("```")[0]
                    .split("{", 1)[-1]
                )
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    req.buffer["prompt"] = prompt
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_DISCUSSION_CHECK_ERROR
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
                if len(resp_dict["team"]) != team_size:
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    err_msg = f"Team size not matched. We need a team with {team_size} players, but received {resp_dict['team']}."
                    LOGGER.debug(err_msg)
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
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    err_msg = f"Duplicate members found on the team. We need a team with {team_size} unique players, but received {resp_dict['team']}."
                    LOGGER.debug(err_msg)
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
                    not isinstance(mem, int) or mem < 0 or mem >= 5
                    for mem in resp_dict["team"]
                ):
                    err_msg = f"Proposed team contains invalid player ids: {resp_dict['team']}. Max player id is 4."
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    LOGGER.debug(err_msg)
                    messages = req.buffer["msg"]
                    messages.append({"role": "assistant", "content": req.resp})
                    messages.append(
                        {
                            "role": "user",
                            "content": PROPOSE_TEAM_INVALID_PLAYER_PROMPT.replace(
                                "{max_player_id}",
                                "4",
                            ),
                        }
                    )
                    req.buffer["msg"] = messages
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                elif any(k not in resp_dict for k in ["rationale", "team"]):
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.TEAM_PROPOSAL_CHECK_ERROR
                    keys = []
                    if "rationale" not in resp_dict:
                        keys.append("rationale")
                    if "team" not in resp_dict:
                        keys.append("team")
                    err_msg = f"{keys} should be included in your answer."
                    LOGGER.debug(err_msg)
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
                    return prompt, RequestStatus.TEAM_PROPOSAL_SUCCEED
        else:
            raise ValueError(f"Unknown status: {req.status}")

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
        role_name: str = req.history["roles"][req.player_idx][1]
        if req.status == RequestStatus.ROLE_GUESS_GET_PROMPT:
            if "tgt_player_i" in req.args:
                player_i = req.args["tgt_player_i"]
                tgt_role: str = req.history["roles"][player_i][1]
            elif role_name == "Servant":
                # randomly pick another player
                player_i = self.seeder.choice(
                    [i for i in range(5) if i != req.player_idx]
                )
                tgt_role: str = req.history["roles"][player_i][1]
                # sample a false role
                if self.seeder.random() < 0.5:
                    tgt_roles = set(
                        [
                            "Merlin",
                            "Servant",
                            "Assassin",
                            "Minion",
                        ]
                    ) - set([tgt_role])
                    tgt_role = self.seeder.choice(list(tgt_roles))
            elif role_name in [
                "Assassin",
                "Minion",
            ]:
                player_i, tgt_role = self.seeder.choice(
                    [
                        (i, role[1])
                        for i, role in enumerate(req.history["roles"])
                        if role[1] in ["Servant", "Merlin"]
                    ]
                )
                if self.seeder.random() < 0.5:
                    tgt_role = "Merlin" if tgt_role == "Servant" else "Servant"
            else:
                raise ValueError("Merlin should not guess other's role.")

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
                if role_name == AvalonBasicConfig.ROLES_REVERSE["Servant"]:
                    prompt += " " + GUESS_ALL_ROLE_PROMPT.replace(
                        "{i}", str(player_i)
                    )
                    included_roles = ["Merlin", "Servant", "Minion"]

                elif role_name in [
                    AvalonBasicConfig.ROLES_REVERSE["Assassin"],
                    AvalonBasicConfig.ROLES_REVERSE["Minion"],
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
                resp_dict: Dict[str, str] = json.loads(
                    "{"
                    + req.resp.split("```json")[-1]
                    .split("```")[0]
                    .split("{", 1)[-1]
                )
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    req.buffer["prompt"] = prompt
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
                        return prompt, 1
                    elif "rationale" not in resp_dict:
                        req.buffer["trial"] += 1
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
                                RequestStatus.ROLE_GUESS_CHECK_ERRORkj,
                            )
                        err_msg = "Your response should provide an integer score from 1 to 10."
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
        if "tgt_player_i" in req.args:
            tgt_player_i = req.args["tgt_player_i"]
        else:
            tgt_player_i = self.seeder.choice(
                [i for i in range(5) if i != req.player_idx]
            )
        tgt_role = self.seeder.choice(["Merlin", "Servant", "Minion"])
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
                resp_dict: Dict[str, str] = json.loads(
                    "{"
                    + req.resp.split("```json")[-1]
                    .split("```")[0]
                    .split("{", 1)[-1]
                )
            except json.JSONDecodeError:
                req.buffer["trial"] += 1
                if req.buffer["trial"] >= self.max_trials:
                    LOGGER.debug(
                        f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                    )
                    req.buffer["msg"] = req.buffer["msg"][:1]
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    req.buffer["prompt"] = prompt
                    req.buffer["trial"] = 0
                    prompt = self._get_prompt_from_msg(req.buffer["msg"])
                    return prompt, RequestStatus.ROLE_BELIEF_CHECK_ERROR
                err_msg = f"`{req.resp}` can't be parsed as JSON. Trial: {req.buffer['trial']}"
                LOGGER.debug(err_msg)
                messages = req.buffer["msg"]
                messages.append({"role": "assistant", "content": req.resp})
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
                req.buffer["msg"] = messages
                prompt = self._get_prompt_from_msg(req.buffer["msg"])
                return prompt, RequestStatus.ROLE_BELIEF_CHECK_ERROR
            else:
                if resp_dict["score"] < 1 or resp_dict["score"] > 10:
                    req.buffer["trial"] += 1
                    if req.buffer["trial"] >= self.max_trials:
                        LOGGER.debug(
                            f"Maximum number of trials ({self.max_trials}) reached. Restart the prompt."
                        )
                        req.buffer["msg"] = req.buffer["msg"][:1]
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        req.buffer["prompt"] = prompt
                        req.buffer["trial"] = 0
                        prompt = self._get_prompt_from_msg(req.buffer["msg"])
                        return prompt, RequestStatus.ROLE_BELIEF_CHECK_ERROR
                    req.buffer["trial"] += 1
                    messages = req.buffer["msg"]
                    err_msg = f"score must be from 1 to 10. Received: {resp_dict['score']}."
                    LOGGER.debug(err_msg)
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
