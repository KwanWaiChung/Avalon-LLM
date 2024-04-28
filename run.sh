temp=0.5
python gameplay.py \
    --model_name meta-llama/Meta-Llama-3-70B-Instruct \
    --inference_strategy_name anyscale \
    --temperature $temp \
    --output_path outputs/llama3-70b_v0.1_temp=${temp}.jsonl \
    --add_strategy_in_history \
    --to_guess_role \
    --to_guess_belief \
    --resume \
    --tokenizer_path meta-llama/Meta-Llama-3-8B-Instruct \
    --max_input_length 7600 \
    --n_games 1000
