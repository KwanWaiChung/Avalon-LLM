import json
import random
import os
from avalonbench_dev.avalon.engine import (
    AvalonGameEnvironment,
    AvalonBasicConfig,
)
from src.server.tasks.avalon.agents.baseline_agents import find_naive_agent
from src.server.tasks.avalon.agents.my_llm_agent import MyLLMAgent
from src.utils.logger import get_logger
from typing import List, Dict, Any, Union

SEED = 111
DISCUSSION = True
NUM_LLM_PLAYERS = 5
INFERENCE_STRATEGY_NAME = "anyscale"
# MODEL_NAME = "google/gemma-7b-it"
MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"
# INFERENCE_STRATEGY_NAME = "together"
# MODEL_NAME = "meta-llama/Llama-3-70b-chat-hf"
END_TOKENS = ["assistant"]
OUTPUT_PATH = "outputs/mixtral-8x22B_v0.1.jsonl"
TEMPERATURE = 0

SEEDER = random.Random(SEED)
LOGGER = get_logger(__name__, logger_level="debug", console_level="debug")
N_GAMES = 20

presets = json.load(open("data/avalon/dev.json"))
# debug
# preset = {
#     "num_players": 5,
#     "quest_leader": 0,
#     # "role_names": ["Assassin", "Merlin", "Servant", "Servant", "Minion"],
#     "role_names": ["Merlin", "Assassin", "Servant", "Servant", "Minion"],
# }

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
games = []
for game_i in range(N_GAMES):
    preset = SEEDER.choice(presets)
    env = AvalonGameEnvironment.from_presets(preset)
    pos = list(range(5))
    llm_pos = SEEDER.sample(pos, k=NUM_LLM_PLAYERS)
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
    }

    for i, (role_i, role_name, side) in enumerate(env.get_roles()):
        if i in llm_pos:
            # init llm
            player_list.append(
                MyLLMAgent(
                    history=history,
                    model_name=MODEL_NAME,
                    inference_strategy_name=INFERENCE_STRATEGY_NAME,
                    name=f"Player {i}",
                    config=env.config,
                    id=i,
                    side=side,
                    seed=SEED,
                    role=role_i,
                    end_tokens=END_TOKENS,
                    temperature=TEMPERATURE,
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
                    discussion=DISCUSSION,
                    seed=SEED,
                )
            )
        # If the player is Merlin or Evil, let them see the sides of all players.
        player_sides = [side for _, _, side in env.get_roles()]
        if player_list[i].role == 0 or player_list[i].side == 0:
            player_list[i].see_sides(player_sides)
            player_list[i].initialize_game_info(player_list=env.get_roles())
        else:
            player_list[i].initialize_game_info(player_list=[])

    # game loop
    while not env.done:
        LOGGER.info(f"Game number: {game_i+1}. Round: {env.turn+1}.")
        phase: int = env.get_phase()[0]
        # team selection phase
        if phase == 0:
            leader = env.get_quest_leader()
            history["leaders"].append(leader)
            LOGGER.info(
                f"Team selection phase, the leader is Player {leader}."
            )
            if DISCUSSION:
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
            ].propose_team(team_size=env.get_team_size(), mission_id=env.turn)
            env.choose_quest_team(team=frozenset(team["team"]), leader=leader)
            history["team_props"].append(team)
        # vote team (vote_on_team)
        elif phase == 1:
            team = env.get_current_quest_team()
            LOGGER.info(
                f"Team vote phase, Player {leader} chose the team {team}."
            )
            votes = []
            for player in player_list:
                vote: Dict[str, Union[str, bool]] = player.vote_on_team(
                    mission_id=env.turn, team=team
                )
                votes.append(vote)

            approved_votes = sum([v["vote"] for v in votes])
            # (next phase, game is done, team is accepted)
            result = env.gather_team_votes([v["vote"] for v in votes])
            history["team_votes"].append(
                {"votes": votes, "result": result[-1]}
            )
            LOGGER.info(
                f"{approved_votes} approved, {len(votes) - approved_votes} failed. The team is {'accepted' if result[-1] else 'failed'}."
            )
        elif phase == 2:
            # vote quest (vote_on_mission)
            quest_team = env.get_current_quest_team()
            LOGGER.info(f"Quest vote phase, the team is {quest_team}.")
            votes = []
            for player in quest_team:
                vote: Dict[str, Union[str, bool]] = player_list[
                    player
                ].vote_on_mission(mission_id=env.turn, quest_team=quest_team)
                votes.append(vote)
            approved_votes = sum([v["vote"] for v in votes])
            # (next phase, game is done, quest succeed?)
            result = env.gather_quest_votes([v["vote"] for v in votes])
            history["quest_votes"].append(
                {"votes": votes, "result": result[-2]}
            )
            num_failed = result[-1]
            LOGGER.info(
                f"{len(votes) - num_failed} approved, {num_failed} failed. The quest {'suceeds' if result[-2] else 'fails'}."
            )
        # assassination phase
        elif phase == 3:
            assassin = env.get_assassin()
            LOGGER.info(f"Assassination phase. Player {assassin} will choose.")
            target = player_list[assassin].assassinate(env.num_players)
            # (next phase, game is done, good wins?)
            result = env.choose_assassination_target(
                assassin, target["merlin"]
            )
            LOGGER.info(
                f"The assassination is {'successful' if result[-1] else 'failed'}."
            )
            history["assassin"] = {
                "target": target["merlin"],
                "success": result[-1],
            }
    # one game has done
    history["final_result"] = env.good_victory
    history["id"] = game_i + 1
    games.append(history)
    LOGGER.info(
        f"Game {game_i+1} is finished. {'Good' if env.good_victory else 'Evil'} wins."
    )
    with open(OUTPUT_PATH, "a+") as f:
        f.write(json.dumps(history, ensure_ascii=False))
