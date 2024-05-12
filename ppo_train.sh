accelerate launch --config_file configs/train/deepspeed.yaml \
  ppo_train.py \
  --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
  --filename "outputs/llama3-70b_v0.1_temp=0.5.jsonl" \
  --lora_path "/kfdata05/kf_grp/wckwan/LLaMA-Factory/saves/Llama-3-8B-Instruct-avalon/lora/sft/checkpoint-1400/" \
  --mini_batch_size 64 \
  --save_path "saves/avalon_ppo"