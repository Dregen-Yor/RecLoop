#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
START_CYCLE=${START_CYCLE:-1}


log() { echo "[$(date +%T)] $1"; }
error() { echo "[ERROR] $1" >&2; exit 1; }
success() { echo "[SUCCESS] $1"; }


DATASET_NAME=${DATASET_NAME:-"Toys_and_Games"}
BACKBONE=${BACKBONE:-"OneRec_0.5B"}
CF_MODEL_NAME=${CF_MODEL_NAME:-"sasrec"}
TEXT_MODE=${TEXT_MODE:-false} # True if use the original index.json, False if use the index.json of the CF model
ITEMS_PER_CYCLE=${ITEMS_PER_CYCLE:-5}
MAX_CYCLES=${MAX_CYCLES:-15}
MAX_CONCURRENT_USERS=${MAX_CONCURRENT_USERS:-100}
# STORAGE_DIR=${STORAGE_DIR:-"$PROJECT_ROOT/simulation_storage/simulation_storage_$(date +%Y%m%d_%H%M%S)_generative_{$DATASET_NAME}_{$BACKBONE}"}
STORAGE_DIR=${STORAGE_DIR:-"$PROJECT_ROOT/simulation_storage/{$DATASET_NAME}_{$BACKBONE}_noexclude_CID"}
RECOMMENDATION_FILE=${RECOMMENDATION_FILE:-"$STORAGE_DIR/recommendations_list_{$DATASET_NAME}_{$BACKBONE}.txt"}
ENABLE_TRAINING=${ENABLE_TRAINING:-true}
PYTHON="your_conda_path_to_recloop/bin/python"
log "Starting single-cycle simulation - Dataset:$DATASET_NAME Users:$NUM_USERS Items:$ITEMS_PER_CYCLE Concurrent users:$MAX_CONCURRENT_USERS"


check_file() { [[ -e "$1" ]] || error "Missing file: $1"; }

log "Checking dependency files..."
check_file "$SCRIPT_DIR/user_profiles_$DATASET_NAME"

NUM_USERS=${NUM_USERS:-$(find "$SCRIPT_DIR/user_profiles_$DATASET_NAME" -type f | wc -l)}
# NUM_USERS=${NUM_USERS:-10}
check_file "$PROJECT_ROOT/simulation/generative_recommend_onerec.sh"


log "Creating storage directory: $STORAGE_DIR"
mkdir -p "$STORAGE_DIR"
mkdir -p "$STORAGE_DIR/output"


LOG_FILE="$STORAGE_DIR/run_simulation_round_generative_onerec.log"
exec > >(tee -a "$LOG_FILE") 2>&1


DATA_FILE="$PROJECT_ROOT/recommenders/tranditionalrec/data/$DATASET_NAME/$DATASET_NAME.txt"
BACKUP_FILE="$STORAGE_DIR/backup_data.txt"

if [[ -f "$DATA_FILE" ]]; then
    cp "$DATA_FILE" "$BACKUP_FILE"
    log "Data file has been backed up"
fi


cp "$SCRIPT_DIR/run_simulation_round_generative_onerec.sh" "$STORAGE_DIR/" && log "Script has been backed up to $STORAGE_DIR"

cp "$SCRIPT_DIR/generative_recommend_onerec.sh" "$STORAGE_DIR/" 2>/dev/null || true
log "✓ Script has been backed up to $STORAGE_DIR"


log "Backing up OneRec metadata..."
mkdir -p "$STORAGE_DIR/onerec_metadata"
ONEREC_DATA_PATH="$PROJECT_ROOT/recommenders/generativerec_onerec/data/$DATASET_NAME"
if [ "$TEXT_MODE" == true ]; then
    cp "$ONEREC_DATA_PATH/${DATASET_NAME}.index.json" "$STORAGE_DIR/onerec_metadata/"
    log "index.jsonhas been backed up"
else
    cp "$ONEREC_DATA_PATH/${DATASET_NAME}-${CF_MODEL_NAME}.index.json" "$STORAGE_DIR/onerec_metadata/"
    log "index.jsonhas been backed up"
fi

if [[ -f "$ONEREC_DATA_PATH/${DATASET_NAME}.item.json" ]]; then
    cp "$ONEREC_DATA_PATH/${DATASET_NAME}.item.json" "$STORAGE_DIR/onerec_metadata/"
    log "item.jsonhas been backed up"
fi

if [[ -d "$ONEREC_DATA_PATH/info" ]]; then
    cp -r "$ONEREC_DATA_PATH/info" "$STORAGE_DIR/onerec_metadata/"
    log "info directoryhas been backed up"
fi

log "OneRec metadataBacked up to $STORAGE_DIR/onerec_metadata/"


cp "$DATA_FILE" "$STORAGE_DIR/data/$DATASET_NAME-1.txt" && log "Initial data generated at $STORAGE_DIR/data/$DATASET_NAME-1.txt"

# cp -r "$PROJECT_ROOT/simulation/user_memory_$DATASET_NAME" "$STORAGE_DIR/user_memory_$DATASET_NAME"

log "Starting simulation..."
cd "$PROJECT_ROOT"

CMD="$PYTHON simulation/closed_loop_recommendation_system.py \
    --dataset $DATASET_NAME \
    --num_users $NUM_USERS \
    --backbone $BACKBONE \
    --items_per_cycle $ITEMS_PER_CYCLE \
    --max_cycles $MAX_CYCLES \
    --start_cycle $START_CYCLE \
    --storage_dir $STORAGE_DIR \
    --recommendation_file $RECOMMENDATION_FILE \
    --recommend_script $PROJECT_ROOT/simulation/generative_recommend_onerec.sh \
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