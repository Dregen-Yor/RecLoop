 
DATASET_NAME=$1
BACKBONE=$2
STORAGE_DIR=$3
CYCLE=$4
NUM_USERS=$5

MODEL_PATH=$STORAGE_DIR/output/${DATASET_NAME}_tiger_bs=4096_lr=0.001_cycle=${CYCLE}.pth
HISTORY_FILE=$STORAGE_DIR/recommendation_history_list_{$DATASET_NAME}_{$BACKBONE}.txt


if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists: $MODEL_PATH，skipping training"
else
    echo "Starting trainingModel..."
    CUDA_VISIBLE_DEVICES=0 python3 ./recommenders/generativerec_tiger_old/model/main.py \
       --data_dir $STORAGE_DIR/data/ \
       --output_dir $STORAGE_DIR/output \
       --cycle $CYCLE \
       --data_name "$DATASET_NAME" > "$STORAGE_DIR/output/${BACKBONE}-${DATASET_NAME}-${CYCLE}.txt"
fi


echo "Aggregating recommendation history..."
python3 -c "
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
echo "Recommendation history aggregation completed: $HISTORY_FILE"

CUDA_VISIBLE_DEVICES=0 python3 ./recommenders/generativerec_tiger_old/model/recommend_gen.py \
   --model_path $MODEL_PATH \
   --data_path $STORAGE_DIR/data/$DATASET_NAME-${CYCLE}.txt \
   --code_path $STORAGE_DIR/data/${DATASET_NAME}_t5_rqvae.npy \
   --output_path $STORAGE_DIR/recommendations_list_{$DATASET_NAME}_{$BACKBONE}.txt \
   --k 5 \
   --cycle $CYCLE \
   --end_user $NUM_USERS \
   --prob_output_dir $STORAGE_DIR/probabilities_list_{$DATASET_NAME}_{$BACKBONE}/ \
   --recommendation_history_file $HISTORY_FILE > "$STORAGE_DIR/recommend.log"
