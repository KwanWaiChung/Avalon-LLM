import json
import os
import re
from typing import Dict, List, Union
from strictfire import StrictFire
from transformers import AutoTokenizer


def get_team_disc(
    history, round_i: int, player_i: int
) -> Dict[str, Union[str, int]]:
    role: str = history["roles"][player_i][1]
    # team_diss
    prompt = history["team_discs"][round_i][player_i]["prompt"]
    resp: str = json.dumps(
        {
            "strategy": history["team_discs"][round_i][player_i]["strategy"],
            "response": history["team_discs"][round_i][player_i]["response"],
        },
        indent=4,
        ensure_ascii=False,
    )
    output = {
        "instruction": prompt,
        "output": resp,
        "game_id": history["id"],
        "round": round_i,
        "player_id": player_i,
        "player_role": role,
        "phase": "team_disc",
    }
    return output


def get_team_prop(
    history,
    round_i: int,
) -> Dict[str, Union[str, int]]:

    player_i: int = history["leaders"][round_i]
    role: str = history["roles"][player_i][1]
    prompt: str = history["team_props"][round_i]["prompt"]
    resp: str = json.dumps(
        {
            "rationale": history["team_props"][round_i]["rationale"],
            "team": history["team_props"][round_i]["team"],
        },
        indent=4,
        ensure_ascii=False,
    )
    output = {
        "instruction": prompt,
        "output": resp,
        "game_id": history["id"],
        "round": round_i,
        "player_id": player_i,
        "player_role": role,
        "phase": "team_prop",
    }
    return output


def get_team_vote(
    history,
    round_i: int,
    player_i: int,
) -> Dict[str, Union[str, int]]:
    role: str = history["roles"][player_i][1]
    prompt: str = history["team_votes"][round_i]["votes"][player_i]["prompt"]
    resp: str = json.dumps(
        {
            "rationale": history["team_votes"][round_i]["votes"][player_i][
                "rationale"
            ],
            "vote": (
                "approve"
                if history["team_votes"][round_i]["votes"][player_i]["vote"]
                else "reject"
            ),
        },
        indent=4,
        ensure_ascii=False,
    )
    output = {
        "instruction": prompt,
        "output": resp,
        "game_id": history["id"],
        "round": round_i,
        "player_id": player_i,
        "player_role": role,
        "phase": "team_vote",
    }
    return output


def get_quest_vote(
    history,
    round_i: int,
    player_i: int,
) -> Dict[str, Union[str, int]]:
    role: str = history["roles"][player_i][1]
    voter_i: int = history["team_props"][round_i]["team"].index(player_i)

    prompt: str = history["quest_votes"][round_i]["votes"][voter_i]["prompt"]
    resp: str = json.dumps(
        {
            "rationale": history["quest_votes"][round_i]["votes"][voter_i][
                "rationale"
            ],
            "vote": (
                "pass"
                if history["quest_votes"][round_i]["votes"][voter_i]["vote"]
                else "fail"
            ),
        },
        indent=4,
        ensure_ascii=False,
    )
    output = {
        "instruction": prompt,
        "output": resp,
        "game_id": history["id"],
        "round": round_i,
        "player_id": player_i,
        "player_role": role,
        "phase": "quest_vote",
    }
    return output


def get_role_guess(
    history,
    round_i: int,
    player_i: int,
):
    src_player_ids: List[int] = [
        turn["src_player"] for turn in history["role_guess"][round_i]
    ]
    if player_i not in src_player_ids:
        return None
    src_player_i = src_player_ids.index(player_i)
    role: str = history["roles"][player_i][1]
    prompt: str = history["role_guess"][round_i][src_player_i]["prompt"]
    resp: str = json.dumps(
        history["role_guess"][round_i][src_player_i]["output"],
        indent=4,
        ensure_ascii=False,
    )
    output = {
        "instruction": prompt,
        "output": resp,
        "game_id": history["id"],
        "round": round_i,
        "player_id": player_i,
        "player_role": role,
        "phase": "role_guess",
    }
    return output


def get_role_belief(
    history,
    round_i: int,
    player_i: int,
):
    src_player_ids: List[int] = [
        turn["src_player"] for turn in history["role_belief"][round_i]
    ]
    if player_i not in src_player_ids:
        return None
    src_player_i = src_player_ids.index(player_i)
    role: str = history["roles"][player_i][1]
    prompt: str = history["role_belief"][round_i][src_player_i]["prompt"]
    resp: str = json.dumps(
        history["role_belief"][round_i][src_player_i]["output"],
        indent=4,
        ensure_ascii=False,
    )
    output = {
        "instruction": prompt,
        "output": resp,
        "game_id": history["id"],
        "round": round_i,
        "player_id": player_i,
        "player_role": role,
        "phase": "role_belief",
    }
    return output


"""things in out_data
1. prompt
2. resp
3. game_id
4. round
5. player_id
6. player_role
7. phase
    team_disc (OK)
    team_prop (OK)
    team_vote (OK)
    quest_vote (OK)
    assassin
    role_guess
    role_belief
8. prompt_len
"""


def split_prompt(s) -> List[str]:
    """_summary_

    Args:
        s (_type_): _description_

    Returns:
        List[str]: intro + n-2 rounds + instruction.
    """
    return re.split(
        r"(?=#### Round \d+ Discussion)|(?=### Your Instruction)", s
    )


def main(
    fn: str,
    out_fn: str,
    tokenizer_path: str,
    max_input_len: int = float("inf"),
):
    data = [json.loads(row) for row in open(fn)]
    out_data = []
    for row in data:
        if row["status"] != "Finished":
            continue
        good_win = row["final_result"]
        player_ids = [
            i for i, role in enumerate(row["roles"]) if role[2] == good_win
        ]
        n_rounds = len(row["leaders"])
        for round_i in range(n_rounds):
            for player_i in player_ids:
                # team disc
                output = get_team_disc(
                    history=row,
                    round_i=round_i,
                    player_i=player_i,
                )
                out_data.append(output)

                # team prop
                leader: int = row["leaders"][round_i]
                if player_i == leader:
                    output = get_team_prop(history=row, round_i=round_i)
                    out_data.append(output)

                # team vote
                output = get_team_vote(
                    history=row, round_i=round_i, player_i=player_i
                )
                out_data.append(output)

                # quest vote
                if (
                    row["team_votes"][round_i]["result"]
                    and player_i in row["team_props"][round_i]["team"]
                ):
                    output = get_quest_vote(
                        history=row, round_i=round_i, player_i=player_i
                    )
                    out_data.append(output)

                # role guess
                output = get_role_guess(
                    history=row,
                    round_i=round_i,
                    player_i=player_i,
                )
                if output is not None:
                    out_data.append(output)

                # belief guess
                output = get_role_belief(
                    history=row, round_i=round_i, player_i=player_i
                )
                if output is not None:
                    out_data.append(output)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    for output in out_data:
        output["input"] = ""  # unused
        prompt_len = len(tokenizer(output["instruction"])["input_ids"])
        ori_prompt_len = prompt_len
        while prompt_len > max_input_len:
            prompt_list = split_prompt(output["instruction"])
            n_rounds = len(prompt_list) - 2
            output["instruction"] = "".join(prompt_list[:1] + prompt_list[2:])
            prompt_len = len(tokenizer(output["instruction"])["input_ids"])
        if ori_prompt_len > prompt_len:
            print(
                f"Reduced original input length {ori_prompt_len} to {prompt_len}."
            )
        output["prompt_len"] = prompt_len

    max_prompt_len = max(o["prompt_len"] for o in out_data)
    mean_prompt_len = sum(o["prompt_len"] for o in out_data) / len(out_data)
    print(
        f"max_prompt_len: {max_prompt_len}, mean_prompt_len: {mean_prompt_len}"
    )
    print(f"Converted {len(out_data)} data.")

    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    with open(out_fn, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
        print(f"Processed file saved in {out_fn}.")


if __name__ == "__main__":
    StrictFire(main)
