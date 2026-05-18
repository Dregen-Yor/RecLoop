#!/bin/bash




set -e


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


log() { echo "[$(date +%T)] $1"; }
error() { echo "[ERROR] $1" >&2; exit 1; }
success() { echo "[SUCCESS] $1"; }


DATASET_NAME=${DATASET_NAME:-"Office_Products"}
BACKBONE=${BACKBONE:-"Mamba4Rec"}
ITEMS_PER_CYCLE=${ITEMS_PER_CYCLE:-5} # recommendation list size
MAX_CYCLES=${MAX_CYCLES:-15}
START_CYCLE=${START_CYCLE:-1}
REFLECTION_INTERVAL=${REFLECTION_INTERVAL:-5}
MAX_CONCURRENT_USERS=${MAX_CONCURRENT_USERS:-20}
# STORAGE_DIR=${STORAGE_DIR:-"$PROJECT_ROOT/simulation_storage/simulation_storage_$(date +%Y%m%d_%H%M%S)_traditional_{$DATASET_NAME}_{$BACKBONE}"}
STORAGE_DIR=${STORAGE_DIR:-"$PROJECT_ROOT/simulation_storage/simulation_storage_traditional_{$DATASET_NAME}_{$BACKBONE}_noexclude"}

RECOMMENDATION_FILE=${RECOMMENDATION_FILE:-"$STORAGE_DIR/recommendations_list_{$DATASET_NAME}_{$BACKBONE}.txt"}
ENABLE_TRAINING=${ENABLE_TRAINING:-true}

log "Starting single-cycle simulation - Dataset:$DATASET_NAME Users:$NUM_USERS Items:$ITEMS_PER_CYCLE Concurrent users:$MAX_CONCURRENT_USERS Start cycle:$START_CYCLE"


check_file() { [[ -e "$1" ]] || error "Missing file: $1"; }

log "Checking dependency files..."
check_file "$SCRIPT_DIR/user_profiles_$DATASET_NAME"
NUM_USERS=${NUM_USERS:-$(find "$SCRIPT_DIR/user_profiles_$DATASET_NAME" -type f | wc -l)}
# NUM_USERS=${NUM_USERS:-10}
check_file "$PROJECT_ROOT/simulation/traditional_recommend.sh"


# if [ -d "$SCRIPT_DIR/user_memory_$DATASET_NAME" ]; then

# else

#     # echo $DATASET_NAME
#     MEMORY_INIT_CMD="uv run python user_memory_initializer.py \
#         --dataset_path $PROJECT_ROOT/recommenders/tranditionalrec/data/$DATASET_NAME \
#         --dataset_name $DATASET_NAME \
#         --memory_storage_dir $SCRIPT_DIR/user_memory_$DATASET_NAME"
#     if $MEMORY_INIT_CMD; then

#     else

#     fi
# fi


log "Creating storage directory: $STORAGE_DIR"
mkdir -p "$STORAGE_DIR"
mkdir -p "$STORAGE_DIR/output"
mkdir -p "$STORAGE_DIR/data"

LOG_FILE="$STORAGE_DIR/run_simulation_round_traditional.log"
exec > >(tee -a "$LOG_FILE") 2>&1
DATA_FILE="$PROJECT_ROOT/recommenders/data/$DATASET_NAME/$DATASET_NAME.txt"
BACKUP_FILE="$STORAGE_DIR/backup_data.txt"

if [[ -f "$DATA_FILE" ]]; then
    cp "$DATA_FILE" "$BACKUP_FILE"
    log "Data file has been backed up"
fi


cp "$SCRIPT_DIR/run_simulation_round_traditional.sh" "$STORAGE_DIR/" && log "Script has been backed up to $STORAGE_DIR"

cp "$PROJECT_ROOT/recommenders/data/$DATASET_NAME/$DATASET_NAME.txt" "$STORAGE_DIR/data/$DATASET_NAME-1.txt" && log "Backed up to $STORAGE_DIR/data/$DATASET_NAME-1.txt"

# cp -r "$PROJECT_ROOT/simulation/user_memory_$DATASET_NAME" "$STORAGE_DIR/user_memory_$DATASET_NAME"

log "Starting simulation..."
cd "$PROJECT_ROOT"

CMD="uv run simulation/closed_loop_recommendation_system.py \
    --dataset $DATASET_NAME \
    --num_users $NUM_USERS \
    --backbone $BACKBONE \
    --items_per_cycle $ITEMS_PER_CYCLE \
    --max_cycles $MAX_CYCLES \
    --start_cycle $START_CYCLE \
    --reflection_interval $REFLECTION_INTERVAL \
    --storage_dir $STORAGE_DIR \
    --recommendation_file $RECOMMENDATION_FILE \
    --recommend_script $PROJECT_ROOT/simulation/traditional_recommend.sh \
    --max_concurrent_users $MAX_CONCURRENT_USERS"

[[ "$ENABLE_TRAINING" == "false" ]] && CMD="$CMD --no_training"

log "Executing command: $CMD"

START_TIME=$(date +%s)
if $CMD; then
    DURATION=$(( $(date +%s) - START_TIME ))
    success "Simulation completed (elapsed: ${DURATION}s)"


    log "Result summary: $(find "$STORAGE_DIR" -name "*.json" | wc -l) JSON files generated"
    [[ -f "$DATA_FILE" ]] && log "Data update: $(($(wc -l < "$DATA_FILE") - $(wc -l < "$BACKUP_FILE"))) new lines"

else
    error "Simulation run failed"


    [[ -f "$BACKUP_FILE" ]] && cp "$BACKUP_FILE" "$DATA_FILE" && log "Data restored"
    exit 1
fi

log "Script execution completed"