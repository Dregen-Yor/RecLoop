#!/bin/bash
# Usage: run from recommenders/generativerec_onerec/
# Override base model: export SFT_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NCCL_IB_DISABLE=1
SFT_BASE_MODEL="${SFT_BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

for category in "Toys_and_Games"; do
    train_file=$(ls -f ./data/Toys_and_Games/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Toys_and_Games/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Toys_and_Games/test/${category}*11.csv)
    info_file=$(ls -f ./data/Toys_and_Games/info/${category}*.txt)
    echo "${train_file}" "${eval_file}" "${info_file}" "${test_file}"

    torchrun --nproc_per_node 1 \
        sft.py \
        --base_model "${SFT_BASE_MODEL}" \
        --batch_size 1024 \
        --micro_batch_size 16 \
        --train_file "${train_file}" \
        --eval_file "${eval_file}" \
        --output_dir "sft/Toys_and_Games" \
        --wandb_project wandb_proj \
        --wandb_run_name wandb_name \
        --category "${category}" \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path "./data/Toys_and_Games/Toys_and_Games.index.json" \
        --item_meta_path "./data/Toys_and_Games/Toys_and_Games.item.json" \
        --freeze_LLM False
done
