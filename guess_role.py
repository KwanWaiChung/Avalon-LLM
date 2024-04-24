# this code formulates prompt prefix given the history
import json
from typing import Dict
from src.server.tasks.avalon.agents.my_llm_agent import MyLLMAgent
from src.server.tasks.avalon.prompts import INTRODUCTION2, RETRY_JSON_PROMPT
from src.utils.inference import (
    TogetherInferenceStrategy,
    AnyscaleInferenceStrategy,
)
from tqdm import tqdm

inference_strategy = AnyscaleInferenceStrategy()
model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"

data = [
    json.loads(row)
    for row in open("outputs/mixtral-8x22B_v0.2_temp=0.3.jsonl")
]

"Based on the gameplay history, rate the likelihood of Player {player_id} being a member of the Resistance (Merlin or Servant) on a scale of 1 to 10, where 10 is very likely and 1 is very unlikely."
good_role_guess_prompt = """### Instruction
Based on the game play history, rate the likelihood of Player {player_id} of being on the good team (Merlin or Servant) on a scale of 1 to 10, with 10 being very likely while 1 means very unlikely. 
First provide your rationale, then provide the score.
Format your output as JSON:
{
    "rationale": <str>,
    "score": <int, 1 to 10>,
}"""

evil_role_guess_prompt = """### Instruction
Based on the game play history, rate the likelihood of Player {player_id} of being on the good team (Merlin or Servant) on a scale of 1 to 10, with 10 being very likely while 1 means very unlikely. 
First provide your rationale, then provide the score.
Format your output as JSON:
{
    "rationale": <str>,
    "score": <int, 1 to 10>,
}"""

output_path = "outputs/mixtral-8x22B_temp=0.3_role_preds.jsonl"
input_tokens = 0
output_tokens = 0
outputs = []
for row in tqdm(data):
    if row["status"] != "Finished":
        continue
    n_turns = len(row["team_discs"])
    for turn_i in range(n_turns):
        history = {}
        history["leaders"] = row["leaders"][: turn_i + 1]
        history["team_discs"] = row["team_discs"][: turn_i + 1]
        history["team_props"] = row["team_props"][:turn_i]
        history["team_votes"] = row["team_votes"][:turn_i]
        history["quest_votes"] = row["quest_votes"][:turn_i]
        history_str = MyLLMAgent.get_history_str(history)
        prompt_prefix = (
            INTRODUCTION2.replace(
                "You are one of the player in the board game called The Resistance: Avalon.",
                "Here is a gameplay the board game called The Resistance: Avalon. Study it carefully and identity the hidden identies of each player.",
            )
            + "\n\n"
            + history_str
        )
        for player_id in range(5):
            # ground truth
            good = row["roles"][player_id][-1]
            good_prompt = (
                prompt_prefix
                + "\n\n"
                + good_role_guess_prompt.replace("{player_id}", str(player_id))
            )
            evil_prompt = (
                prompt_prefix
                + "\n\n"
                + evil_role_guess_prompt.replace("{player_id}", str(player_id))
            )
            # good (5 players)
            # evil (5 players)
            for prompt, ans in [(good_prompt, good), (evil_prompt, not good)]:
                messages = [{"role": "user", "content": prompt}]
                n_trials = 3
                for i in range(n_trials):
                    try:
                        output = inference_strategy.generate(
                            messages=messages,
                            model_name=model_name,
                            temperature=0,
                            top_p=1,
                            max_tokens=512,
                            end_tokens=["assistant"],
                        )
                        resp: str = output["output"]
                        input_tokens += output["prompt_len"]
                        output_tokens += output["output_len"]
                        resp_dict: Dict[str, str] = json.loads(
                            resp.split("```json")[-1].split("```")[0]
                        )
                    except json.JSONDecodeError:
                        print(
                            f"{resp} can't be parsed as JSON. Trial: {i}/{n_trials}."
                        )
                        messages.append(
                            {"role": "assistant", "content": output["output"]}
                        )
                        messages.append(
                            {"role": "user", "content": RETRY_JSON_PROMPT}
                        )
                    else:
                        break
                output_dict = {
                    "prompt": prompt,
                    "resp": resp_dict,
                    "ans": ans,
                    "id": f"game={row['id']}-player={player_id}-round={turn_i+1}",
                }
                with open(output_path, "a+") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
