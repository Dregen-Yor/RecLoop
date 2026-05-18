#!/bin/bash
# Usage: run from recommenders/generativerec_onerec/
# Override SFT checkpoint: export RL_MODEL_PATH=sft/Toys_and_Games/checkpoint-670

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NCCL_IB_DISABLE=1

for category in "Toys_and_Games"; do
    train_file=$(ls -f ./data/Toys_and_Games/train/${category}*.csv)
    eval_file=$(ls -f ./data/Toys_and_Games/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Toys_and_Games/info/${category}*.txt)

    MODEL_PATH="${RL_MODEL_PATH:-sft/${category}/checkpoint-670}"

    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 2 --main_process_port 29503 \
        rl.py \
        --model_path "$MODEL_PATH" \
        --train_batch_size 128 \
        --eval_batch_size 128 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 2 \
        --train_file "${train_file}" \
        --eval_file "${eval_file}" \
        --info_file "${info_file}" \
        --category "${category}" \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type ranking \
        --num_generations 16 \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir rl \
        --wandb_run_name wandb_name \
        --sid_index_path "./data/Toys_and_Games/Toys_and_Games.index.json" \
        --item_meta_path "./data/Toys_and_Games/Toys_and_Games.item.json"
done
