import json
import random
import os
from src.server.tasks.avalon.engine import (
    AvalonBasicConfig,
    AvalonGameEnvironment,
)
from src.server.tasks.avalon.agents.baseline_agents import find_naive_agent
from src.server.tasks.avalon.agents.my_llm_agent import (
    MyLLMAgent,
    OutputException,
)
from src.utils.logger import get_logger
from typing import List, Dict, Union
from strictfire import StrictFire


def main(
    model_name,
    inference_strategy_name,
    output_path,
    n_games=20,
    seed: int = 111,
    num_llm_players: int = 5,
    end_tokens: List[str] = [],
    temperature=0,
    tokenizer_path: str = None,
    max_input_length: int = None,
    to_discuss=True,
    add_strategy_in_prompt=False,
    to_guess_role: bool = False,
    to_guess_multiple_player_role: bool = True,
    n_guess_role_repeat: int = 1,
    to_guess_belief: bool = False,
    use_summary: bool = False,
    resume: bool = False,
):
    """_summary_

    Args:
        model_name (_type_): _description_
        inference_strategy_name (_type_): _description_
        output_path (_type_): _description_
        n_games (int, optional): _description_. Defaults to 20.
        seed (int, optional): _description_. Defaults to 111.
        num_llm_players (int, optional): _description_. Defaults to 5.
        end_tokens (List[str], optional): _description_. Defaults to [].
        temperature (int, optional): _description_. Defaults to 0.
        tokenizer_path (str): The path of the tokenizer. If provided, it will be used
            to check whether the input exceed max_input_length.
        max_input_length (int): The maximum input length allowed.
        to_discuss (bool, optional): _description_. Defaults to True.
        add_strategy_in_prompt (bool, optional): _description_. Defaults to False.
        to_guess_role (bool): Guess other's role after discussion.
        to_guess_multiple_player_role (bool): Guess multiple player's role simultaneously.
        n_guess_role_repeat (int): Number of samples to generate.
            It should be > 1 for dpo and expert iteration training.
        to_guess_belief (bool): Guess other's belief on your role.
        use_summary (bool): Summarize gameplay as a summary and use it
            in the prompt instead.
        resume (bool): Resume previous session. Note that the random state
            won't be recovered.

    """
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
        console_level="debug",
        file_level="debug",
        log_path=os.path.join("logs", log_name),
    )
    if to_guess_role != to_guess_belief:
        raise ValueError(
            "Current only support guess role and guess belief together."
        )

    game_i = 0
    if resume:
        if os.path.exists(output_path):
            games = [json.loads(row) for row in open(output_path)]
            completed_games = sum(
                [row["status"] == "Finished" for row in games]
            )
            n_games -= completed_games
            game_i = int(games[-1]["id"])
            logger.info(f"Resuming from {output_path}.")
        else:
            logger.info("Specified `to_resume`, but output_path is not found.")

    for _ in range(n_games):
        # Keep sampling if encountered error
        while True:
            preset = seeder.choice(presets)
            env = AvalonGameEnvironment.from_presets(preset)
            pos = list(range(5))
            llm_pos = seeder.sample(pos, k=num_llm_players)
            player_list = []
            history = {
                "leaders": [],
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
                "id": game_i + 1,
            }

            for i, (role_i, role_name, side) in enumerate(env.get_roles()):
                if i in llm_pos:
                    # init llm
                    player_list.append(
                        MyLLMAgent(
                            history=history,
                            model_name=model_name,
                            inference_strategy_name=inference_strategy_name,
                            name=f"Player {i}",
                            config=env.config,
                            id=i,
                            side=side,
                            seed=seed,
                            role=role_i,
                            end_tokens=end_tokens,
                            temperature=temperature,
                            add_strategy_in_prompt=add_strategy_in_prompt,
                            use_summary=use_summary,
                            tokenizer_path=tokenizer_path,
                            max_input_length=max_input_length,
                        )
                    )
                else:
                    # naive players
                    player_list.append(
                        find_naive_agent(
                            id=i,
                            name=f"Player {i}",
                            config=env.config,
                            side=side,
                            role=role_i,
                            num_players=5,
                            session=None,
                            role_name=role_name,
                            merlin=env.config.merlin,
                            percival=env.config.percival,
                            morgana=env.config.morgana,
                            mordred=env.config.mordred,
                            oberon=env.config.oberon,
                            num_good=env.config.num_good,
                            num_evil=env.config.num_evil,
                            discussion=to_discuss,
                            seed=seed,
                        )
                    )
                # If the player is Merlin or Evil, let them see the sides of all players.
                player_sides = [side for _, _, side in env.get_roles()]
                if player_list[i].role == 0 or player_list[i].side == 0:
                    player_list[i].see_sides(player_sides)
                    player_list[i].initialize_game_info(
                        player_list=env.get_roles()
                    )
                else:
                    player_list[i].initialize_game_info(player_list=[])

            # game loop
            while not env.done:
                logger.info(f"Game number: {game_i+1}. Round: {env.turn+1}.")
                phase: int = env.get_phase()[0]
                try:
                    # team selection phase
                    if phase == 0:
                        leader = env.get_quest_leader()
                        history["leaders"].append(leader)
                        logger.info(
                            f"Team selection phase, the leader is Player {leader}."
                        )
                        if to_discuss:
                            # for player in player_list:
                            #     player.summarize()
                            history["team_discs"].append([])
                            for player in player_list:
                                resp = player.team_discussion(
                                    team_size=env.get_team_size(),
                                    team_leader_id=leader,
                                    mission_id=env.turn,
                                )
                                history["team_discs"][-1].append(resp)
                        if to_guess_role:
                            # only ask servant and evil team to guess
                            # servant guess any others.
                            # evil team guess which good player is Merlin.
                            history["role_guess"].append([])
                            for player in player_list:
                                if player.role_name == "Merlin":
                                    continue
                                elif player.role_name == "Servant":
                                    # randomly pick another player
                                    player_i = seeder.choice(
                                        [i for i in range(5) if i != player.id]
                                    )
                                    tgt_role = history["roles"][player_i][1]
                                    # sample a false role
                                    if seeder.random() < 0.5:
                                        tgt_roles = set(
                                            [
                                                "Merlin",
                                                "Servant",
                                                "Assassin",
                                                "Minion",
                                            ]
                                        ) - set([tgt_role])
                                        tgt_role = seeder.choice([tgt_roles])
                                elif player.role_name in [
                                    "Assassin",
                                    "Minion",
                                ]:
                                    player_i = seeder.choice(
                                        [
                                            player.id
                                            for player in player_list
                                            if player.role_name
                                            in ["Servant", "Merlin"]
                                        ]
                                    )
                                    tgt_role = player_list[player_i].role_name
                                    if seeder.random() < 0.5:
                                        tgt_role = (
                                            "Merlin"
                                            if tgt_role == "Servant"
                                            else "Servant"
                                        )
                                else:
                                    raise NotImplementedError(
                                        f"Not implemented for {player.role}."
                                    )
                                resp = player.guess_role(
                                    player_i=player_i,
                                    to_guess_multiple_player=to_guess_multiple_player_role,
                                    role_to_guess=(
                                        None
                                        if to_guess_multiple_player_role
                                        else tgt_role
                                    ),
                                    n_repeat=n_guess_role_repeat,
                                )

                                resp["src_player"] = player.id
                                resp["tgt_player"] = player_i
                                history["role_guess"][-1].append(resp)
                        if to_guess_belief:
                            history["role_belief"].append([])
                            for player in player_list:
                                player_i = seeder.choice(
                                    [i for i in range(5) if i != player.id]
                                )
                                role = seeder.choice(
                                    ["Merlin", "Servant", "Minion"]
                                )
                                resp = player.guess_belief(
                                    player_i=player_i, tgt_role=role
                                )
                                resp["src_player"] = player.id
                                resp["tgt_player"] = player_i
                                resp["tgt_role"] = role
                                history["role_belief"][-1].append(resp)
                        if use_summary:
                            summaries = []
                            for player in player_list:
                                resp = player.summarize()
                                summaries.append(resp)
                            history["summaries"].append(summaries)

                        # after discussion, choose team (propose_team)
                        team: Dict[str, Union[str, List[int]]] = player_list[
                            leader
                        ].propose_team(
                            team_size=env.get_team_size(), mission_id=env.turn
                        )
                        env.choose_quest_team(
                            team=frozenset(team["team"]), leader=leader
                        )
                        history["team_props"].append(team)
                    # vote team (vote_on_team)
                    elif phase == 1:
                        team = env.get_current_quest_team()
                        logger.info(
                            f"Team vote phase, Player {leader} chose the team {team}."
                        )
                        votes = []
                        for player in player_list:
                            vote: Dict[str, Union[str, bool]] = (
                                player.vote_on_team(
                                    mission_id=env.turn, team=team
                                )
                            )
                            votes.append(vote)

                        approved_votes = sum([v["vote"] for v in votes])
                        # (next phase, game is done, team is accepted)
                        result = env.gather_team_votes(
                            [v["vote"] for v in votes]
                        )
                        history["team_votes"].append(
                            {"votes": votes, "result": result[-1]}
                        )
                        logger.info(
                            f"{approved_votes} approved, {len(votes) - approved_votes} failed. The team is {'accepted' if result[-1] else 'failed'}."
                        )
                        # empty dict if rejected. Edited in phase==2 if approved.
                        history["quest_votes"].append({})
                    elif phase == 2:
                        # vote quest (vote_on_mission)
                        quest_team = env.get_current_quest_team()
                        logger.info(
                            f"Quest vote phase, the team is {quest_team}."
                        )
                        votes = []
                        for player in quest_team:
                            vote: Dict[str, Union[str, bool]] = player_list[
                                player
                            ].vote_on_mission(
                                mission_id=env.turn, quest_team=quest_team
                            )
                            votes.append(vote)
                        approved_votes = sum([v["vote"] for v in votes])
                        # (next phase, game is done, quest succeed?)
                        result = env.gather_quest_votes(
                            [v["vote"] for v in votes]
                        )
                        history["quest_votes"][-1] = {
                            "votes": votes,
                            "result": result[-2],
                        }
                        num_failed = result[-1]
                        logger.info(
                            f"{len(votes) - num_failed} approved, {num_failed} failed. The quest {'suceeds' if result[-2] else 'fails'}."
                        )
                    # assassination phase
                    elif phase == 3:
                        assassin = env.get_assassin()
                        logger.info(
                            f"Assassination phase. Player {assassin} will choose."
                        )
                        target = player_list[assassin].assassinate(
                            env.num_players
                        )
                        # (next phase, game is done, good wins?)
                        result = env.choose_assassination_target(
                            assassin, target["merlin"]
                        )
                        logger.info(
                            f"The assassination is {'failed' if result[-1] else 'successful'}."
                        )
                        target["success"] = not result[-1]
                        history["assassin"] = target
                except OutputException as e:
                    logger.error(e)
                    logger.info("Going to restart another game.")
                    history["status"] = "Error"
                    with open(output_path, "a+") as f:
                        f.write(json.dumps(history, ensure_ascii=False) + "\n")
                    break  # break the game loop to sample another game.
            # one game has done
            game_i += 1
            if env.done:
                history["final_result"] = env.good_victory
                history["status"] = "Finished"
                logger.info(
                    f"Game {game_i} is finished. {'Good' if env.good_victory else 'Evil'} wins."
                )
                with open(output_path, "a+") as f:
                    f.write(json.dumps(history, ensure_ascii=False) + "\n")
                break  # break while true loop


if __name__ == "__main__":
    StrictFire(main)
