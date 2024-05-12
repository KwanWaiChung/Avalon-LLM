import json
from tqdm import tqdm
from typing import Dict
from src.server.tasks.avalon.agents.my_llm_agent import MyLLMAgent
from src.server.tasks.avalon.my_prompts import RETRY_JSON_PROMPT
from src.utils.inference.anyscale_strategy import AnyscaleInferenceStrategy

fn = "/kfdata05/kf_grp/wckwan/Avalon-LLM/outputs/llama3-70b_v0.2.{i}_temp=0.5.jsonl"
inference_strategy = AnyscaleInferenceStrategy()
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
for i in tqdm(range(1, 5)):
    data = [json.loads(row) for row in open(fn.format(i=i))]
    for row in data:
        for round_i, role_guesses in enumerate(row["role_guess"]):
            for player_i, role_guess in enumerate(role_guesses):
                # role guess
                prefix = role_guess["prompt"].split("### Game Play History")[0]
                inst = (
                    "### Your Instruction"
                    + role_guess["prompt"].split("### Your Instruction")[1]
                )
                history = {
                    "leaders": row["leaders"][: round_i + 1],
                    "team_discs": row["team_discs"][: round_i + 1],
                    "team_props": row["team_props"][:round_i],
                    "team_votes": row["team_votes"][:round_i],
                    "quest_votes": row["quest_votes"][:round_i],
                    "role_guess": row["role_guess"][:round_i],
                    "role_belief": row["role_belief"][:round_i],
                    "summaries": row["summaries"][:round_i],
                }
                history_str = MyLLMAgent.get_history_str(
                    history=history, use_summary=True, summary_idx=player_i
                )
                prompt = prefix + history_str + "\n\n" + inst
                messages = [{"role": "user", "content": prompt}]
                for j in range(3):
                    try:
                        output = inference_strategy.generate(
                            messages=messages,
                            model_name=model_name,
                            temperature=0.5,
                            max_tokens=512,
                        )
                        resp: str = output["output"]
                        resp_dict: Dict[str, str] = json.loads(
                            "{"
                            + resp.split("```json")[-1]
                            .split("```")[0]
                            .split("{", 1)[-1]
                        )
                    except json.JSONDecodeError:
                        err_msg = f"`{resp}` can't be parsed as JSON. Trial: {j}/{3}."
                        print(err_msg)
                        messages.append(
                            {"role": "assistant", "content": output["output"]}
                        )
                        messages.append(
                            {"role": "user", "content": RETRY_JSON_PROMPT}
                        )
                    else:
                        break
                role_guess["output"] = resp_dict
                role_guess["prompt"] = prompt
        for round_i, role_guesses in enumerate(row["role_belief"]):
            for player_i, role_guess in enumerate(role_guesses):
                # role guess
                prefix = role_guess["prompt"].split("### Game Play History")[0]
                inst = (
                    "### Your Instruction"
                    + role_guess["prompt"].split("### Your Instruction")[1]
                )
                history = {
                    "leaders": row["leaders"][: round_i + 1],
                    "team_discs": row["team_discs"][: round_i + 1],
                    "team_props": row["team_props"][:round_i],
                    "team_votes": row["team_votes"][:round_i],
                    "quest_votes": row["quest_votes"][:round_i],
                    "role_guess": row["role_guess"][:round_i],
                    "role_belief": row["role_belief"][:round_i],
                    "summaries": row["summaries"][:round_i],
                }
                history_str = MyLLMAgent.get_history_str(
                    history=history, use_summary=True, summary_idx=player_i
                )
                prompt = prefix + history_str + "\n\n" + inst
                messages = [{"role": "user", "content": prompt}]
                for j in range(3):
                    try:
                        output = inference_strategy.generate(
                            messages=messages,
                            model_name=model_name,
                            temperature=0.5,
                            max_tokens=512,
                        )
                        resp: str = output["output"]
                        resp_dict: Dict[str, str] = json.loads(
                            "{"
                            + resp.split("```json")[-1]
                            .split("```")[0]
                            .split("{", 1)[-1]
                        )
                    except json.JSONDecodeError:
                        err_msg = f"`{resp}` can't be parsed as JSON. Trial: {j}/{3}."
                        print(err_msg)
                        messages.append(
                            {"role": "assistant", "content": output["output"]}
                        )
                        messages.append(
                            {"role": "user", "content": RETRY_JSON_PROMPT}
                        )
                    else:
                        break
                role_guess["output"] = resp_dict
                role_guess["prompt"] = prompt
    with open(fn.format(i=f"{i}_1"), "w") as f:
        f.write(
            "\n".join([json.dumps(row, ensure_ascii=False) for row in data])
        )
