#!/bin/bash
# Usage: bash generative_recommend_onerec.sh <DATASET_NAME> <BACKBONE> <STORAGE_DIR> <CYCLE> <NUM_USERS>
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DATASET_NAME=$1
BACKBONE=$2
STORAGE_DIR=$3
CYCLE=$4
NUM_USERS=$5
PYTHON="your_conda_path_to_recloop/bin/python"

if [ -z "$DATASET_NAME" ] || [ -z "$BACKBONE" ] || [ -z "$STORAGE_DIR" ] || [ -z "$CYCLE" ] || [ -z "$NUM_USERS" ]; then
    echo "Error: missing required arguments"
    echo "Usage: $0 <DATASET_NAME> <BACKBONE> <STORAGE_DIR> <CYCLE> <NUM_USERS>"
    exit 1
fi

# HuggingFace model IDs by default; set ONEREC_BASE_MODEL_* to local checkpoint dirs if needed
ONEREC_BASE_MODEL_0_5B="${ONEREC_BASE_MODEL_0_5B:-Qwen/Qwen2.5-0.5B-Instruct}"
ONEREC_BASE_MODEL_1_5B="${ONEREC_BASE_MODEL_1_5B:-Qwen/Qwen2.5-1.5B-Instruct}"
ONEREC_BASE_MODEL_3B="${ONEREC_BASE_MODEL_3B:-Qwen/Qwen2.5-3B-Instruct}"

TEXT_MODE=0 # True if use the original index.json, False if use the index.json of the CF model
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-uv run python}"

HISTORY_FILE=$STORAGE_DIR/recommendation_history_list_{$DATASET_NAME}_{$BACKBONE}.txt
if [ "$BACKBONE" == "OneRec" ] || [ "$BACKBONE" == "OneRec_0.5B" ]; then
    BASE_MODEL="$ONEREC_BASE_MODEL_0_5B"
elif [ "$BACKBONE" == "OneRec_3B" ]; then
    BASE_MODEL="$ONEREC_BASE_MODEL_3B"
elif [ "$BACKBONE" == "OneRec_1.5B" ]; then
    BASE_MODEL="$ONEREC_BASE_MODEL_1_5B"
else
    echo "Invalid backbone: $BACKBONE"
    exit 1
fi
# ============================================

# ============================================
echo "=========================================="
echo "Step 1: Convert simulation data to OneRec format"
echo "=========================================="

ONEREC_DATA_DIR=$STORAGE_DIR/onerec_data
mkdir -p $ONEREC_DATA_DIR/train

CURRENT_TXT_FILE=$STORAGE_DIR/data/$DATASET_NAME-${CYCLE}.txt
if [ "$TEXT_MODE" == 0 ]; then
    TRAIN_CSV=$ONEREC_DATA_DIR/train/${DATASET_NAME}_cycle_${CYCLE}_train.csv
    VALID_CSV=$ONEREC_DATA_DIR/train/${DATASET_NAME}_cycle_${CYCLE}_valid.csv
    TEST_CSV=$ONEREC_DATA_DIR/train/${DATASET_NAME}_cycle_${CYCLE}_test.csv
else
    TRAIN_CSV=$ONEREC_DATA_DIR/train/${DATASET_NAME}_cycle_${CYCLE}_train_CID.csv
    VALID_CSV=$ONEREC_DATA_DIR/train/${DATASET_NAME}_cycle_${CYCLE}_valid_CID.csv
    TEST_CSV=$ONEREC_DATA_DIR/train/${DATASET_NAME}_cycle_${CYCLE}_test_CID.csv
fi

echo "Current data: $CURRENT_TXT_FILE"
echo "Train CSV: $TRAIN_CSV"
echo "Validation CSV: $VALID_CSV"
echo "Test CSV: $TEST_CSV"
echo "Text_mode: $TEXT_MODE"
$PYTHON ./recommenders/generativerec_onerec/simulation_data_converter.py \
    --current_txt $CURRENT_TXT_FILE \
    --onerec_data_dir ./recommenders/generativerec_onerec/data/$DATASET_NAME \
    --output_train_csv $TRAIN_CSV \
    --output_valid_csv $VALID_CSV \
    --output_test_csv $TEST_CSV \
    --dataset_name $DATASET_NAME \
    --Text_mode $TEXT_MODE \
    > "$STORAGE_DIR/logs/data_converter.log" 2>&1

if [ $? -ne 0 ]; then
    echo "✗ Data conversion failed"
    exit 1
fi

echo "✓ Data conversion completed"

# ============================================

# ============================================
echo ""
echo "=========================================="
echo "Step 2: SFT training"
echo "=========================================="

MODEL_PATH=$STORAGE_DIR/output/checkpoint-cycle-${CYCLE}


if [ "$TEXT_MODE" == true ]; then
    SID_INDEX_PATH=$STORAGE_DIR/onerec_metadata/${DATASET_NAME}.index.json
    INFO_FILE=$(ls ./recommenders/generativerec_onerec/data/$DATASET_NAME/info/${DATASET_NAME}*.txt | head -n 1)
else
    SID_INDEX_PATH=$STORAGE_DIR/onerec_metadata/${DATASET_NAME}-sasrec.index.json
    INFO_FILE=$(ls ./recommenders/generativerec_onerec/data/$DATASET_NAME/info/${DATASET_NAME}-*CID.txt | head -n 1)
fi
ITEM_META_PATH=$STORAGE_DIR/onerec_metadata/${DATASET_NAME}.item.json


if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model already exists: $MODEL_PATH，skipping training"
else
    echo "Starting SFT training..."
    
    $PYTHON ./recommenders/generativerec_onerec/sft_simulation.py \
        --data_dir $ONEREC_DATA_DIR \
        --output_dir $STORAGE_DIR/output \
        --cycle $CYCLE \
        --data_name $DATASET_NAME \
        --base_model $BASE_MODEL \
        --sid_index_path $SID_INDEX_PATH \
        --item_meta_path $ITEM_META_PATH \
        > "$STORAGE_DIR/output/sft-${DATASET_NAME}-${CYCLE}.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "✗ SFT training failed, check log: $STORAGE_DIR/output/sft-${DATASET_NAME}-${CYCLE}.log"
        tail -n 50 "$STORAGE_DIR/output/sft-${DATASET_NAME}-${CYCLE}.log"
        exit 1
    fi
    
    echo "✓ SFT training completed"
fi


echo ""
echo "=========================================="
echo "Step 3: Aggregate recommendation history"
echo "=========================================="

$PYTHON -c "
import os
history = {}
for i in range(1, $CYCLE):
    rec_file = '$STORAGE_DIR/recommendations_list_{$DATASET_NAME}_{$BACKBONE}_' + str(i) + '.txt'
    if os.path.exists(rec_file):
        with open(rec_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = parts[0]
                    items = parts[1:]
                    if user_id not in history:
                        history[user_id] = []
                    history[user_id].extend(items)
with open('$HISTORY_FILE', 'w') as f:
    for user_id in sorted(history.keys(), key=int):
        f.write(user_id + ' ' + ' '.join(history[user_id]) + '\n')
print(f'Aggregation completed: {len(history)} users, from {CYCLE - 1} cycles')
"

echo "✓ Recommendation history aggregation completed: $HISTORY_FILE"

echo ""
echo "=========================================="
echo "Step 4: Generate recommendations"
echo "=========================================="

echo "Info file: $INFO_FILE"
echo "Model path: $MODEL_PATH"
echo "Data path: $CURRENT_TXT_FILE (read from txt)"
echo "Item metadata path: $ITEM_META_PATH"
mkdir -p $STORAGE_DIR/logs/recommend/
$PYTHON ./recommenders/generativerec_onerec/recommend_gen_onerec.py \
    --model_path $MODEL_PATH \
    --data_path $CURRENT_TXT_FILE \
    --info_file $INFO_FILE \
    --index_file $SID_INDEX_PATH \
    --item_meta_path $ITEM_META_PATH \
    --output_path $STORAGE_DIR/recommendations_list_{$DATASET_NAME}_{$BACKBONE}.txt \
    --k 5 \
    --cycle $CYCLE \
    --end_user $NUM_USERS \
    --batch_size 1 \
    --num_beams 20 \
    --category $DATASET_NAME \
    --recommendation_history_file $HISTORY_FILE \
    > "$STORAGE_DIR/logs/recommend/recommend_cycle_${CYCLE}.log" 2>&1

if [ $? -ne 0 ]; then
    echo "✗ Recommendation generation failed, check log: $STORAGE_DIR/logs/recommend/recommend_cycle_${CYCLE}.log"
    tail -n 50 "$STORAGE_DIR/logs/recommend/recommend_cycle_${CYCLE}.log"
    exit 1
fi

echo "✓ Recommendation generation completed"
