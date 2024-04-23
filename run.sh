temp=0.3
python gameplay.py \
    --model_name mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --inference_strategy_name anyscale \
    --temperature $temp \
    --output_path outputs/mixtral-8x22B_v0.2_temp=${temp}.jsonl 