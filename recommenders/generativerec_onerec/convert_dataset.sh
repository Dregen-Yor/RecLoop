#!/bin/bash


PYTHON_SCRIPT="convert_dataset.py"

DATASET_NAME="Office_Products"
INPUT_DIR="data/$DATASET_NAME"

OUTPUT_DIR="data/$DATASET_NAME"

# ===========================================

echo "Start converting $DATASET_NAME ..."

python $PYTHON_SCRIPT \
    --dataset_name $DATASET_NAME \
    --data_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --category $DATASET_NAME \
    --CID 1 \
    --seed 42

echo "Finished!"
