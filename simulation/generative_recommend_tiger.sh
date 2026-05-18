#!/bin/bash

# ============================================================================

# 




#

#   bash generative_recommend_tiger.sh <DATASET_NAME> <BACKBONE> <STORAGE_DIR> <CYCLE> <NUM_USERS>
#






# ============================================================================

set -e


DATASET_NAME=$1
BACKBONE=$2
STORAGE_DIR=$3
CYCLE=$4
NUM_USERS=$5
TEST_MODE=${TEST_MODE:-"false"} # True if use the original index.json, False if use the index.json of the CF model

if [ -z "$DATASET_NAME" ] || [ -z "$BACKBONE" ] || [ -z "$STORAGE_DIR" ] || [ -z "$CYCLE" ] || [ -z "$NUM_USERS" ]; then
    echo "Error: missing required arguments"
    echo "Usage: $0 <DATASET_NAME> <BACKBONE> <STORAGE_DIR> <CYCLE> <NUM_USERS>"
    exit 1
fi


MODEL_DIR=$STORAGE_DIR/output/${DATASET_NAME}_cycle_${CYCLE}
HISTORY_FILE=$STORAGE_DIR/recommendation_history_list_{$DATASET_NAME}_{$BACKBONE}.txt
DATA_FILE=$STORAGE_DIR/data/$DATASET_NAME-${CYCLE}.txt
if [ "$TEST_MODE" == "True" ]; then
    SEMANTIC_ID_FILE=$STORAGE_DIR/data/${DATASET_NAME}_sentence-t5-xxl_42.pkl
else
    SEMANTIC_ID_FILE=$STORAGE_DIR/data/${DATASET_NAME}_SASRec_42.pkl
fi
OUTPUT_FILE=$STORAGE_DIR/recommendations_list_{$DATASET_NAME}_{$BACKBONE}.txt

echo "============================================================================"
echo "TIGER single-cycle training and recommendation generation"
echo "============================================================================"
echo "Dataset: $DATASET_NAME"
echo "Model: $BACKBONE"
echo "Cycle: $CYCLE"
echo "User count: $NUM_USERS"
echo "Storage directory: $STORAGE_DIR"
echo "============================================================================"
echo ""

# ============================================

# ============================================
echo "=========================================="
echo "Step 1: Train TIGER model"
echo "=========================================="


if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/results/ckpt_best.pt" ]; then
    echo "✓ Model already exists: $MODEL_DIR，skipping training"
else
    echo "Starting training..."
    echo "  - Data file: $DATA_FILE"
    echo "  - Semantic ID file: $SEMANTIC_ID_FILE"
    echo "  - Output directory: $MODEL_DIR"
    echo ""
    

    if [ ! -f "$DATA_FILE" ]; then
        echo "✗ Error: data file does not exist: $DATA_FILE"
        exit 1
    fi
    
    if [ ! -f "$SEMANTIC_ID_FILE" ]; then
        echo "⚠ Warning: semantic ID file does not exist: $SEMANTIC_ID_FILE, running rq-vae automatically"
        # exit 1
    fi
    

    mkdir -p "$STORAGE_DIR/output"
    


    CUDA_VISIBLE_DEVICES=1 uv run python ./recommenders/generativerec/run_simulation.py \
        --cycle $CYCLE \
        --data_dir $STORAGE_DIR/data \
        --data_name $DATASET_NAME \
        --output_dir $STORAGE_DIR/output \
        --device_id 0 \
        --seed 42 \
        --dataset_type amazon \
        --content_model sentence-t5-xxl \
        --features_needed title,price,brand,categories \
        --prompt_format amazon \
        --flag_use_learnable_text_embed \
        --text_embedding_dim 4096 \
        --hidden_sizes 4096,2048 \
        --embed_proj_type mlp \
        --embd_proj_in_dropout_rate 0.1 \
        --embd_proj_dropout_rate 0.1 \
        --learning_rate 5e-4 \
        --weight_decay 0.0 \
        --num_train_epochs 30 \
        --per_device_train_batch_size 256 \
        --per_device_eval_batch_size 256 \
        --gradient_accumulation_steps 1 \
        --warmup_ratio 0.1 \
        --logging_steps 100 \
        --eval_steps 500 \
        --save_steps 500 \
        --max_grad_norm 1.0 \
        --eval_accumulation_steps 1 \
        --log_mode disabled \
        > "$STORAGE_DIR/output/train_cycle_${CYCLE}.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "✗ Training failed, check log: $STORAGE_DIR/output/train_cycle_${CYCLE}.log"
        tail -n 50 "$STORAGE_DIR/output/train_cycle_${CYCLE}.log"
        exit 1
    fi
    
    echo "✓ Training completed"
fi

echo ""

# ============================================

# ============================================
echo "=========================================="
echo "Step 2: Aggregating recommendation history"
echo "=========================================="

if [ $CYCLE -eq 1 ]; then
    echo "Cycle 1, no history aggregation needed"

    touch "$HISTORY_FILE"
else
    echo "Aggregating recommendation history from cycle 1 to cycle $((CYCLE-1))..."
    
    python3 -c "
import os
history = {}
for i in range(1, $CYCLE):
    rec_file = '$STORAGE_DIR/recommendations_list_${DATASET_NAME}_${BACKBONE}_' + str(i) + '.txt'
    if os.path.exists(rec_file):
        print(f'  Reading cycle {i}: {rec_file}')
        with open(rec_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = parts[0]
                    items = parts[1:]
                    if user_id not in history:
                        history[user_id] = []
                    history[user_id].extend(items)
    else:
        print(f'  Recommendation file for cycle {i} does not exist: {rec_file}')

with open('$HISTORY_FILE', 'w') as f:
    for user_id in sorted(history.keys(), key=int):
        f.write(user_id + ' ' + ' '.join(history[user_id]) + '\n')

print(f'✓ Aggregation completed: {len(history)} users, from {$CYCLE - 1} cycles')
"
    
    if [ $? -ne 0 ]; then
        echo "✗ Recommendation history aggregation failed"
        exit 1
    fi
fi

echo "✓ Recommendation history file: $HISTORY_FILE"
echo ""

# ============================================

# ============================================
echo "=========================================="
echo "Step 3: Generate recommendations"
echo "=========================================="

echo "Model path: $MODEL_DIR"
echo "Data file: $DATA_FILE"
echo "Semantic ID file: $SEMANTIC_ID_FILE"
echo "Output files: $OUTPUT_FILE"
echo "Recommendation history file: $HISTORY_FILE"
echo "Recommendation count: 5"
echo "User range: 1-$NUM_USERS"
echo ""


mkdir -p "$STORAGE_DIR/logs"


CUDA_VISIBLE_DEVICES=1 uv run python ./recommenders/generativerec/recommend_gen.py \
    --model_path $MODEL_DIR \
    --data_path $DATA_FILE \
    --semantic_id_path $SEMANTIC_ID_FILE \
    --output_path $OUTPUT_FILE \
    --k 5 \
    --cycle $CYCLE \
    --start_user 1 \
    --end_user $NUM_USERS \
    --beam_size 20 \
    --max_items_per_seq 20 \
    --codebook_size 256 \
    --exclude_seen \
    --recommendation_history_file $HISTORY_FILE \
    --num_layers 6 \
    --num_decoder_layers 6 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --dropout_rate 0.2 \
    --feed_forward_proj relu \
    --n_positions 258 \
    --initializer_factor 0.02 \
    --seed 42 \
    > "$STORAGE_DIR/logs/recommend_cycle_${CYCLE}.log" 2>&1

if [ $? -ne 0 ]; then
    echo "✗ Recommendation generation failed, check log: $STORAGE_DIR/logs/recommend_cycle_${CYCLE}.log"
    tail -n 50 "$STORAGE_DIR/logs/recommend_cycle_${CYCLE}.log"
    exit 1
fi

echo "✓ Recommendation generation completed"
echo ""

# ============================================

# ============================================
echo "============================================================================"
echo "✓ Cycle $CYCLE processing completed"
echo "============================================================================"
echo "Output files:"
echo "  - Model: $MODEL_DIR"
echo "  - Recommendation list: $OUTPUT_FILE"
echo "  - Recommendation backup: ${OUTPUT_FILE%.txt}_${CYCLE}.txt"
echo "  - Recommendation history: $HISTORY_FILE"
echo "  - Training log: $STORAGE_DIR/output/train_cycle_${CYCLE}.log"
echo "  - Recommendation log: $STORAGE_DIR/logs/recommend_cycle_${CYCLE}.log"
echo "============================================================================"
echo ""

exit 0
