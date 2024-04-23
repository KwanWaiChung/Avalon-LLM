import json
import random
import os
from avalonbench_dev.avalon.engine import (
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
    do_discussion=True,
    num_llm_players: int = 5,
    end_tokens: List[str] = [],
    temperature=0,
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
    games = []
    for game_i in range(n_games):
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
                            discussion=do_discussion,
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
                        if do_discussion:
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
                        history["quest_votes"].append(
                            {"votes": votes, "result": result[-2]}
                        )
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
                    games.append(history)
                    with open(output_path, "a+") as f:
                        f.write(json.dumps(history, ensure_ascii=False) + "\n")
                    break  # break the game loop to sample another game.
            # one game has done
            if env.done:
                history["final_result"] = env.good_victory
                history["status"] = "Finished"
                games.append(history)
                logger.info(
                    f"Game {game_i+1} is finished. {'Good' if env.good_victory else 'Evil'} wins."
                )
                with open(output_path, "a+") as f:
                    f.write(json.dumps(history, ensure_ascii=False) + "\n")
                break  # break while true loop


if __name__ == "__main__":
    logger = get_logger(__name__, logger_level="debug", console_level="debug")
    StrictFire(main)
