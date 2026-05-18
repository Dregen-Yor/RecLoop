#!/bin/bash

STORAGE_DIR=$1
DATASET_NAME=$2
CYCLE=$3
GPU_ID=${4:-0}


if [ -z "$STORAGE_DIR" ] || [ -z "$DATASET_NAME" ] || [ -z "$CYCLE" ]; then
    echo "Error: missing required arguments"
    echo "Usage: bash example_eval.sh <STORAGE_DIR> <DATASET_NAME> <CYCLE> [GPU_ID]"
    echo "Example: bash example_eval.sh /path/to/storage Toys_and_Games 1 0"
    exit 1
fi


MODEL_PATH="$STORAGE_DIR/output/checkpoint-cycle-$CYCLE"
DATA_DIR="$STORAGE_DIR/onerec_data"
OUTPUT_DIR="$STORAGE_DIR/output"


echo "=================================="
echo "OneRec Single-cycle evaluation"
echo "=================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Dataset: $DATASET_NAME"
echo "Cycle: $CYCLE"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT_DIR"
echo "=================================="
echo ""


if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ Model path does not exist: $MODEL_PATH"
    exit 1
fi


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


uv run python "$SCRIPT_DIR/eval_single_model.py" \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --data_name "$DATASET_NAME" \
    --cycle $CYCLE \
    --output_dir "$OUTPUT_DIR" \
    --gpu_id $GPU_ID \
    --batch_size 8 \
    --num_beams 50 \
    --max_new_tokens 256 \
    --length_penalty 0.0

if [ $? -ne 0 ]; then
    echo "✗ Evaluation failed"
    exit 1
fi

echo ""
echo "=================================="
echo "Evaluation completed！"
echo "=================================="
