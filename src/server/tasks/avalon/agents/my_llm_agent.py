from typing import List, Literal, Dict, Any, Union
from avalonbench_dev.avalon.engine import AvalonBasicConfig
from src.server.tasks.avalon.agents.agent import Agent
from src.server.tasks.avalon.prompts import (
    CHOOSE_TEAM_LEADER,
    INFO_ROLE,
    INFO_YOUR_ROLE,
    INTRODUCTION,
    INTRODUCTION2,
    REVEAL_PROMPTS,
    SUMMARIZE,
    TEAM_DISCUSSION,
    TEAM_DISCUSSION2,
    TEAM_DISCUSSION3,
    TEAM_DISCUSSION4,
    PROPOSE_TEAM_PROMPT,
    RETRY_JSON_PROMPT,
    PROPOSE_TEAM_INVALID_SIZE_PROMPT,
    PROPOSE_TEAM_INVALID_PLAYER_PROMPT,
    VOTE_MISSION_ACTION2,
    TEAM_VOTE,
    ASSASSINATION_PROMPT,
)
from src.utils.inference import (
    OpenAIInferenceStrategy,
    TogetherInferenceStrategy,
    AnyscaleInferenceStrategy,
)
import json
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class OutputException(Exception):
    pass


class MyAgentBase(Agent):
    def __init__(self, history: Dict[str, Any], *args, **kwargs):
        self.history = history
        super().__init__(*args, **kwargs)

    def sumamrize(self) -> str:
        raise NotImplementedError

    def propose_team(self, mission_id: int, team_size: int) -> frozenset[int]:
        raise NotImplementedError


class MyLLMAgent(MyAgentBase):
    def __init__(
        self,
        history: Dict[str, Any],
        model_name: str,
        inference_strategy_name: Literal["openai", "together", "anyscale"],
        name: str,
        config: AvalonBasicConfig,  # just pass env.config.
        id: int,
        role: int,
        side=None,
        seed=None,
        temperature: float = 0.1,
        top_p: float = 1,
        max_tokens: int = 512,
        end_tokens: List[str] = [],
        to_recommend_strategy: bool = False,
        add_strategy_in_history: bool = False,
        **kwargs,
    ):
        """
        Initialize the agent with the given parameters.

        Args:
            history (Dict[str, Any]): The game history book keeping object.
            model_name (str): The name of the language model.
            inference_strategy_name (Literal["openai", "together", "anyscale"]):
                The name of the inference strategy.
            name (str): The name of the agent.
            config (AvalonBasicConfig): The game configuration.
            id (int): The id of the agent.
            role (int): The role of the agent.
            side (Optional[int]): The side of the agent (1 for good, 0 for evil).
            seed (Optional[int]): The seed for the random number generator.
            temperature (float): The temperature for the language model.
            top_p (float): The top-p for the language model.
            max_tokens (int): The maximum number of tokens for the language model.
            end_tokens (List[str]): The end tokens for the language model.
            to_recommend_strategy (bool): Whether to prompt the recommended strategy.
            add_strategy_in_history (bool): Add the self-generated strategy in history.

        """
        super().__init__(
            history=history,
            id=id,
            role=role,
            config=config,
        )
        self.model_name = model_name
        if inference_strategy_name == "openai":
            self.inference_strategy = OpenAIInferenceStrategy()
        elif inference_strategy_name == "together":
            self.inference_strategy = TogetherInferenceStrategy()
        elif inference_strategy_name == "anyscale":
            self.inference_strategy = AnyscaleInferenceStrategy()
        else:
            raise OutputException(
                f"Unrecognized strategy: {inference_strategy_name}."
            )
        self.name = name
        self.side = side  # 1 for good, 0 for evil
        self.discussion = kwargs.pop("discussion", None)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.end_tokens = end_tokens
        self.to_recommend_strategy = to_recommend_strategy
        self.add_strategy_in_history = add_strategy_in_history

    def see_sides(self, sides):
        self.player_sides = sides

    def initialize_game_info(self, player_list) -> None:
        """Initiliaze the game info for the agent, which includes game introduction, role, and reveal information for different roles."""
        # Introduction Prompt
        verbal_side = ["Evil", "Good"]
        intro_prompt = INTRODUCTION2
        intro_prompt += "\n"

        minion = ""
        servant_list = []
        assassin = ""
        merlin = ""
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
                raise OutputException(f"Unrecognized role: {player_info[1]}")

        self.identity_prompt = (
            f"You are {self.name}, with identity {self.role_name}."
        )

        self.reveal_prompt = ""
        good_team = sorted(servant_list + [merlin])
        if self.role_name == "Merlin":
            self.reveal_prompt = (
                f"You know that Players {minion} and {assassin} are Evil, and Players"
                f" {', '.join(servant_list)} are Servants."
            )
        elif self.role_name == "Minion":
            self.reveal_prompt = (
                f"You know that Player {assassin} is Assassin, and Players"
                f" {', '.join(good_team)} are on the good team, but you do not know who is Merlin."
            )
        elif self.role_name == "Assassin":
            self.reveal_prompt = (
                f"You know that Player {minion} is Minion, and Players"
                f" {', '.join(good_team)} are on the good team, but you do not know who is Merlin."
            )

        self.system_info = intro_prompt.strip()

    def vote_on_team(
        self, mission_id: int, team: frozenset[int]
    ) -> Dict[str, Union[str, bool]]:
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

        prompt = (
            self._get_prompt_prefix()
            + " "
            + TEAM_VOTE.replace("{team}", ", ".join([str(p) for p in team]))
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        n_trials = 3
        for i in range(n_trials):
            try:
                output = self.inference_strategy.generate(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    end_tokens=self.end_tokens,
                )
                resp: str = output["output"]
                self.history["input_tokens"] += output["prompt_len"]
                self.history["output_tokens"] += output["output_len"]
                resp_dict: Dict[str, str] = json.loads(
                    resp.split("```json")[-1].split("```")[0]
                )
            except json.JSONDecodeError:
                err_msg = (
                    f"{resp} can't be parsed as JSON. Trial: {i}/{n_trials}."
                )
                LOGGER.debug(err_msg)
                messages.append(
                    {"role": "assistant", "content": output["output"]}
                )
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
            else:
                if resp_dict["vote"] not in ["approve", "reject"]:
                    messages.append(
                        {"role": "assistant", "content": output["output"]}
                    )
                    error_msg = f"The vote should be either `approve` or `reject`, but you provided `{resp_dict['vote']}`."
                    messages.append({"role": "user", "content": error_msg})
                else:
                    resp_dict["vote"] = resp_dict["vote"] == "approve"

                    break
        else:
            raise OutputException(err_msg)
        LOGGER.info(
            f"LLM Agent (Player {self.id}, Role: {self.role_name}) voted: {resp_dict['vote']}"
        )
        resp_dict["prompt"] = prompt
        return resp_dict

    def vote_on_mission(
        self, mission_id: int, quest_team: frozenset[int]
    ) -> bool:
        r"""Vote on a quest (team).

        Args:
            mission_id (int): The id of the mission. num_fails = self.config.num_fails_for_quest[mission_id]
            quest_team (frozenset[int]): The list of player ids included in the quest.

        Returns:
            bool: The vote result.
        """
        prompt = (
            self._get_prompt_prefix()
            + " "
            + VOTE_MISSION_ACTION2.replace(
                "{team}", ", ".join([str(p) for p in quest_team])
            )
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        n_trials = 3
        for i in range(n_trials):
            try:
                output = self.inference_strategy.generate(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    end_tokens=self.end_tokens,
                )
                resp: str = output["output"]
                self.history["input_tokens"] += output["prompt_len"]
                self.history["output_tokens"] += output["output_len"]
                resp_dict: Dict[str, str] = json.loads(
                    resp.split("```json")[-1].split("```")[0]
                )
            except json.JSONDecodeError:
                err_msg = (
                    f"{resp} can't be parsed as JSON. Trial: {i}/{n_trials}."
                )
                LOGGER.debug(err_msg)
                messages.append(
                    {"role": "assistant", "content": output["output"]}
                )
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
            else:
                if resp_dict["vote"] not in ["pass", "fail"]:
                    messages.append(
                        {"role": "assistant", "content": output["output"]}
                    )
                    error_msg = f"The vote should be either `pass` or `fail`, but you provided `{resp_dict['vote']}`."
                    messages.append({"role": "user", "content": error_msg})
                else:
                    resp_dict["vote"] = resp_dict["vote"] == "pass"
                    LOGGER.info(
                        f"LLM Agent (Player {self.id}, Role: {self.role_name}) voted: {resp_dict['vote']}"
                    )
                    break
        else:
            raise OutputException(err_msg)
        resp_dict["prompt"] = prompt
        return resp_dict

    def assassinate(self, num_players: int) -> int:
        r"""Assassinate a player.

        Args:
            num_players (int): The number of players in the game.

        Returns:
            int: The id of the player to assassinate. The id is in the range [0, num_players).
        """
        prompt = (
            self._get_prompt_prefix()
            + "\n\n"
            + ASSASSINATION_PROMPT.replace("{max_player_id}", str(num_players))
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        n_trials = 3
        for i in range(n_trials):
            try:
                output = self.inference_strategy.generate(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    end_tokens=self.end_tokens,
                )
                resp: str = output["output"]
                self.history["input_tokens"] += output["prompt_len"]
                self.history["output_tokens"] += output["output_len"]
                resp_dict: Dict[str, str] = json.loads(
                    resp.split("```json")[-1].split("```")[0]
                )
            except json.JSONDecodeError:
                err_msg = (
                    f"{resp} can't be parsed as JSON. Trial: {i}/{n_trials}."
                )
                LOGGER.debug(err_msg)
                messages.append(
                    {"role": "assistant", "content": output["output"]}
                )
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
            else:
                if (
                    resp_dict["merlin"] < 0
                    or resp_dict["merlin"] >= num_players
                ):
                    err_msg = f"Invalid player id: {resp_dict['merlin']}. Max player id is {self.config.num_players-1}"
                    LOGGER.debug(err_msg)
                    messages.append(
                        {"role": "assistant", "content": output["output"]}
                    )
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
                else:
                    break
        else:
            raise OutputException(err_msg)
        LOGGER.info(
            f"LLM Agent (Player {self.id}, Role: {self.role_name}) voted Player {resp_dict['merlin']} as Merlin."
        )
        resp_dict["prompt"] = prompt
        return resp_dict

    def get_believed_sides(self, num_players: int) -> List[float]:
        r"""Get the believed sides of all players.

        Args:
            num_players (int): The number of players in the game.

        Returns:
            List[float]: The list of believed sides (probability) of all players.
        """
        raise NotImplementedError

    def summarize(self) -> str:
        messages = [
            {
                "role": "user",
                "content": self.system_info + "\n" + SUMMARIZE,
            }
        ]
        output = self.inference_strategy.generate(
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        resp: str = output["output"]
        self.history["input_tokens"] += output["prompt_len"]
        self.history["output_tokens"] += output["output_len"]
        return resp

    @staticmethod
    def get_history_str(
        history: Dict[str, Any], strategy_idx: int = None
    ) -> str:
        output = ["### Game Play History"]
        n_round = len(history["leaders"])
        for i in range(n_round):
            # history.append(f"Leader is Player {history['leaders'][i]}")
            if any(resp for resp in history["team_discs"][i]):
                output.append(f"\n#### Round {i + 1} Discussion")
                if strategy_idx is not None and strategy_idx < len(
                    history["team_discs"][i]
                ):
                    output.append(
                        f"**Strategy:** {history['team_discs'][i][strategy_idx]['strategy']}"
                    )
                for p_i, resp in enumerate(history["team_discs"][i]):
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
                    vote["vote"] for vote in history["team_votes"][i]["votes"]
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
                    vote["vote"] for vote in history["quest_votes"][i]["votes"]
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

    def _get_prompt_prefix(self):
        history_str = self.get_history_str(
            self.history, self.id if self.add_strategy_in_history else None
        )
        prompt = self.system_info + "\n\n" + history_str

        prompt += (
            "\n\n### Your Instruction\n"
            + self.identity_prompt
            + " "
            + self.reveal_prompt
        )
        return prompt.strip()

    def team_discussion(self, team_size, team_leader_id, mission_id):
        prompt = self._get_prompt_prefix()
        if self.id == team_leader_id:
            prompt += " You are the Quest leader of this round."
        else:
            prompt += (
                f" Player {team_leader_id} is the Quest leader of this round."
            )
        prompt += f" This Quest requires {team_size} players to vote."

        # history.append(f"Leader is Player {self.history['leaders'][i]}")
        if not any(resp for resp in self.history["team_discs"][-1]):
            prompt += " You are the first to speak in this round."
        prompt += " " + TEAM_DISCUSSION4

        # json error handling
        messages = [{"role": "user", "content": prompt}]
        n_trials = 3
        for i in range(n_trials):
            try:
                output = self.inference_strategy.generate(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    end_tokens=self.end_tokens,
                )
                resp: str = output["output"]
                self.history["input_tokens"] += output["prompt_len"]
                self.history["output_tokens"] += output["output_len"]
                resp: Dict[str, str] = json.loads(
                    resp.split("```json")[-1].split("```")[0]
                )
            except json.JSONDecodeError:
                LOGGER.debug(
                    f"{resp} can't be parsed as JSON. Trial: {i}/{n_trials}."
                )
                messages.append(
                    {"role": "assistant", "content": output["output"]}
                )
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
            else:
                break
        else:
            raise OutputException(
                f"{resp} can't be parsed as JSON despite {n_trials} attempts."
            )
        LOGGER.info(
            f"LLM Agent (Player {self.id}, Role: {self.role_name}): {resp['response']}"
        )
        resp["prompt"] = prompt
        return resp

    def propose_team(
        self, team_size, mission_id
    ) -> Dict[str, Union[str, List[int]]]:
        """
        Propose a team for a mission.

        Args:
            team_size (int): The desired size of the team.
            mission_id (int): The ID of the mission.

        Raises:
            OutputException: If the proposed team is invalid
                (e.g., team size mismatch, invalid player IDs).

        Returns:
            A dictionary with the following keys:
                `rationale` (str): The rationale for the proposed team.
                `team` (List[int]): The list of player IDs in the proposed team.
        """
        prompt = self._get_prompt_prefix()
        prompt += " " + PROPOSE_TEAM_PROMPT.replace(
            "{num_player}", str(team_size)
        ).replace("{max_player_id}", str(self.config.num_players - 1))

        # json error handling
        messages = [{"role": "user", "content": prompt}]
        n_trials = 3
        for i in range(n_trials):
            try:
                output = self.inference_strategy.generate(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    end_tokens=self.end_tokens,
                )
                self.history["input_tokens"] += output["prompt_len"]
                self.history["output_tokens"] += output["output_len"]
                resp: str = output["output"]
                resp_dict: Dict[str, str] = json.loads(
                    resp.split("```json")[-1].split("```")[0]
                )
            except json.JSONDecodeError:
                err_msg = (
                    f"{resp} can't be parsed as JSON. Trial: {i}/{n_trials}."
                )
                LOGGER.debug(err_msg)
                messages.append(
                    {"role": "assistant", "content": output["output"]}
                )
                messages.append({"role": "user", "content": RETRY_JSON_PROMPT})
            else:
                if len(resp_dict["team"]) != team_size:
                    err_msg = f"Team size not matched. We need a team with {team_size} players, but received {resp_dict['team']}."
                    LOGGER.debug(err_msg)
                    messages.append(
                        {"role": "assistant", "content": output["output"]}
                    )
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
                elif any(
                    mem < 0 or mem >= self.config.num_players
                    for mem in resp_dict["team"]
                ):
                    err_msg = f"Proposed team contains invalid player ids: {resp_dict['team']}. Max player id is {self.config.num_players-1}"
                    LOGGER.debug(err_msg)
                    messages.append(
                        {"role": "assistant", "content": output["output"]}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": PROPOSE_TEAM_INVALID_PLAYER_PROMPT.replace(
                                "{max_player_id}",
                                str(self.config.num_players - 1),
                            ),
                        }
                    )
                else:
                    break
        else:
            raise OutputException(err_msg)
        resp_dict["prompt"] = prompt
        return resp_dict
