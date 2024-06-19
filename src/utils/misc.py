import rootutils


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

import torch
import numpy as np
import random
import os
import json
from typing import Any, Tuple, List, Dict
from colorama import Fore, Style
from src.server.tasks.avalon.my_prompts import (
    ASSASSIN_STRATEGY,
    INTRODUCTION,
    MERLIN_REVEAL_PROMPT,
    EVIL_REVEAL_PROMPT,
    MERLIN_STRATEGY,
    MINION_STRATEGY,
    MORGANA_STRATEGY,
    PERCIVAL_REVEAL_PROMPT,
    PERCIVAL_STRATEGY,
    SERVANT_STRATEGY,
    SIX_PLAYERS_SETTING,
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
    QUEST_VOTE_STRATEGY,
)

seeder = random.Random(233)


def get_project_root() -> str:
    return os.environ["PROJECT_ROOT"]
    # p = __file__
    # for _ in range(3):
    #     p = os.path.dirname(p)
    # return os.path.abspath(p)


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def print_count_parameters(model: torch.nn.Module):
    """
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
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"All params: {all_param}, trainable params: {trainable_params}, trainable% {100* trainable_params / all_param}"
    )


def format_messages(messages: List[Dict[str, str]]) -> str:
    """Formats the messages in a conversation with different colors based on role.

    Args:
        messages (List[Dict[str, str]]): A list of dictionaries representing
            messages in the conversation.  Each dictionary should have 'role'
            and 'content' keys. The 'role' can be 'system', 'user', or 'assistant',
            and the 'content' contains the text of the message.

    """
    s = ""
    for msg in messages:
        if msg["role"] == "system":
            s += f"{Fore.BLUE} {Style.BRIGHT} System: {msg['content']} {Style.RESET_ALL}\n"
        elif msg["role"] == "user":
            s += f"{Fore.CYAN} {Style.BRIGHT} User: {msg['content']} {Style.RESET_ALL}\n"
        elif msg["role"] == "assistant":
            s += f"{Fore.MAGENTA} {Style.BRIGHT} Assistant: {msg['content']} {Style.RESET_ALL}\n"
    return s


def parse_json(resp: str):
    resp_dict: Dict[str, str] = json.loads(
        "{"
        + resp.split("```json")[-1]
        .split("```")[0]
        .split("{", 1)[-1]
        .split("}", 1)[0]
        + "}"
    )
    return resp_dict


def get_player_str(player_list: List[int]) -> str:
    if len(player_list) == 2:
        player_str = " and ".join(
            [f"Player {player}" for player in player_list]
        )
    else:
        player_str = ", ".join([f"Player {player}" for player in player_list])
        player_str = (
            player_str.rsplit(", ", 1)[0]
            + ", and "
            + player_str.rsplit(", ", 1)[-1]
        )
    return player_str


def get_game_info_prompt(
    player_list,
    player_id,
    add_strategy_in_prompt=False,
    add_quest_strategy_in_prompt=False,
) -> None:
    """Initiliaze the game info for the agent, which includes game introduction, role, and reveal information for different roles."""
    # Introduction Prompt
    verbal_side = ["Evil", "Good"]
    num_players = len(player_list)
    players_settings = {6: SIX_PLAYERS_SETTING}

    intro_prompt = INTRODUCTION.replace(
        "{player settings}", players_settings[num_players]
    )
    intro_prompt += "\n"

    assassin = None
    merlin = None
    minion = None
    servant = None
    percival = None
    morgana = None

    evil_players_info_for_minion = ""
    evil_players_info_for_assassin = ""
    evil_players_info_for_morgana = ""
    name = f"Player {player_id}"
    role_name = player_list[player_id][1]
    good_team: List[int] = []
    evil_team: List[int] = []
    for idx, player_info in enumerate(player_list):
        if player_info[1] == "Minion":
            minion = str(idx)
            evil_team.append(idx)
            evil_players_info_for_assassin += f"Player {idx} is Minion"
            evil_players_info_for_morgana += f"Player {idx} is Minion"
        elif player_info[1] == "Servant":
            good_team.append(idx)
        elif player_info[1] == "Merlin":
            merlin = idx
        elif player_info[1] == "Assassin":
            evil_team.append(idx)
            evil_players_info_for_minion += f"Player {idx} is Assassin"
            evil_players_info_for_morgana += f"Player {idx} is Assassin"
        elif player_info[1] == "Morgana":
            morgana = idx
            evil_team.append(idx)
            evil_players_info_for_minion += f"Player {idx} is Morgana"
            evil_players_info_for_assassin += f"Player {idx} is Morgana"
        elif player_info[1] == "Percival":
            percival = idx
            good_team.append(idx)
        else:
            raise ValueError(f"Unrecognized role: {player_info[1]}")

    identity_prompt = f"You are {name}, with identity {role_name}."

    reveal_prompt = ""
    if role_name == "Merlin":
        evil_team_str = get_player_str(evil_team)
        good_team_str = get_player_str(good_team)
        reveal_prompt = MERLIN_REVEAL_PROMPT.replace(
            "{evil_players}", evil_team_str
        ).replace("{good_players}", good_team_str)
        strategy = MERLIN_STRATEGY
    elif role_name == "Minion":
        good_team_str = get_player_str(good_team + [merlin])
        reveal_prompt = EVIL_REVEAL_PROMPT.replace(
            "{evil_players_info}", evil_players_info_for_minion
        ).replace("{good_players}", good_team_str)
        strategy = MINION_STRATEGY
    elif role_name == "Assassin":
        good_team_str = get_player_str(good_team + [merlin])
        reveal_prompt = EVIL_REVEAL_PROMPT.replace(
            "{evil_players_info}", evil_players_info_for_assassin
        ).replace("{good_players}", good_team_str)
        strategy = ASSASSIN_STRATEGY
    elif role_name == "Morgana":
        good_team_str = get_player_str(good_team + [merlin])
        reveal_prompt = EVIL_REVEAL_PROMPT.replace(
            "{evil_players_info}", evil_players_info_for_morgana
        ).replace("{good_players}", good_team_str)
        strategy = MORGANA_STRATEGY
    elif role_name == "Percival":
        players = [morgana, merlin]
        seeder.shuffle(players)
        players_str = get_player_str(players)
        reveal_prompt = PERCIVAL_REVEAL_PROMPT.replace(
            "{players}", players_str
        )
        strategy = PERCIVAL_STRATEGY
    elif role_name == "Servant":
        strategy = SERVANT_STRATEGY
    else:
        raise ValueError(f"Unrecognized role name: {role_name}.")

    system_info = intro_prompt.strip()
    if add_strategy_in_prompt:
        reveal_prompt += " " + strategy
    if add_quest_strategy_in_prompt:
        reveal_prompt += " " + QUEST_VOTE_STRATEGY

    return system_info, identity_prompt, reveal_prompt.strip()


def format_history(
    history: Dict[str, Any],
    strategy_idx: int = None,
    n_rounds_to_skip: int = 0,
    summary_idx: int = None,
    use_summary: bool = False,
) -> str:
    output = ["### Game Play History"]
    n_round = len(history["leaders"])
    start_round = 0

    # make sure summary_idx would be str if it needs to be
    if (
        isinstance(summary_idx, int)
        and history["summaries"]
        and history['summaries'][0]
        and isinstance(list(history["summaries"][0].keys())[0], str)
    ):
        summary_idx = str(summary_idx)

    if (
        use_summary
        and history["summaries"]
        and summary_idx in history["summaries"][0]
    ):
        #
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
        output.append(history["summaries"][start_round][summary_idx]["resp"])

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
            if (
                strategy_idx is not None
                and strategy_idx in history["team_discs"][i]
            ):
                output.append(
                    f"**Strategy:** {history['team_discs'][i][strategy_idx]['strategy']}"
                )
            for p_i, resp in history["team_discs"][i].items():
                if resp and resp["response"].strip():
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

        if (
            i < len(history["team_votes"])
            and history["team_votes"][i]["result"] is not None
        ):
            output.append(f"\n#### Round {i+1} Team Votes")
            if (
                strategy_idx is not None
                and strategy_idx in history["team_votes"][i]["votes"]
            ):
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
            and history["quest_votes"][i]["result"] is not None
        ):
            output.append(f"\n#### Round {i+1} Quest Votes")
            if (
                strategy_idx is not None
                and strategy_idx in history["quest_votes"][i]["votes"]
            ):
                output.append(
                    f"**Strategy:** {history['quest_votes'][i]['votes'][strategy_idx]['rationale']}"
                )

            # 'votes' doesn't exist in human data
            if "votes" in history["quest_votes"][i]:
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
