#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


log() { echo "[$(date +%T)] $1"; }
error() { echo "[ERROR] $1" >&2; exit 1; }
success() { echo "[SUCCESS] $1"; }

DATASET_NAME=${DATASET_NAME:-"Toys_and_Games"}
BACKBONE=${BACKBONE:-"TIGER"}
TEST_MODE=${TEST_MODE:-"false"} # True if use the original index.json, False if use the index.json of the CF model
ITEMS_PER_CYCLE=${ITEMS_PER_CYCLE:-5}
MAX_CYCLES=${MAX_CYCLES:-15}
START_CYCLE=${START_CYCLE:-5}
MAX_CONCURRENT_USERS=${MAX_CONCURRENT_USERS:-100}
STORAGE_DIR=${STORAGE_DIR:-"$PROJECT_ROOT/simulation_storage/{$DATASET_NAME}_{$BACKBONE}_noexclude_CID"}
RECOMMENDATION_FILE=${RECOMMENDATION_FILE:-"$STORAGE_DIR/recommendations_list_{$DATASET_NAME}_{$BACKBONE}.txt"}
ENABLE_TRAINING=${ENABLE_TRAINING:-true}


NUM_USERS=${NUM_USERS:-$(find "$SCRIPT_DIR/user_profiles_$DATASET_NAME" -type f 2>/dev/null | wc -l)}
if [ "$NUM_USERS" -eq 0 ]; then
    error "Cannot determine user count: user_profiles_$DATASET_NAME directory is missing or empty"
fi

log "============================================================================"
log "TIGER simulation loop started"
log "============================================================================"
log "Dataset: $DATASET_NAME"
log "Model: $BACKBONE"
log "User count: $NUM_USERS"
log "Items per cycle: $ITEMS_PER_CYCLE"
log "Max cycles: $MAX_CYCLES"
log "Max concurrent users: $MAX_CONCURRENT_USERS"
log "Storage directory: $STORAGE_DIR"
log "Start cycle: $START_CYCLE"
log "Training mode: $([ "$ENABLE_TRAINING" == "true" ] && echo "enabled" || echo "disabled")"
log "============================================================================"


check_file() { [[ -e "$1" ]] || error "Missing file: $1"; }

log "Checking dependency files..."
check_file "$SCRIPT_DIR/user_profiles_$DATASET_NAME"
check_file "$PROJECT_ROOT/simulation/generative_recommend_tiger.sh"
check_file "$PROJECT_ROOT/simulation/closed_loop_recommendation_system.py"

log "✓ Dependency file check completed"
log ""

# ============================================

# ============================================
log "Step 1: Create storage directory"
mkdir -p "$STORAGE_DIR"
mkdir -p "$STORAGE_DIR/output"
mkdir -p "$STORAGE_DIR/data"
mkdir -p "$STORAGE_DIR/logs"
log "✓ Storage directory created: $STORAGE_DIR"
log ""

# ============================================

# ============================================
LOG_FILE="$STORAGE_DIR/run_simulation_round_generative_tiger.log"
exec > >(tee -a "$LOG_FILE") 2>&1
log "✓ Logs will be written to console and file: $LOG_FILE"
log ""

# ============================================

# ============================================
log "Step 2: Back up data and scripts"


DATA_FILE="$PROJECT_ROOT/recommenders/data/$DATASET_NAME/$DATASET_NAME.txt"
BACKUP_FILE="$STORAGE_DIR/backup_data.txt"

if [[ -f "$DATA_FILE" ]]; then
    cp "$DATA_FILE" "$BACKUP_FILE"
    log "✓ Original data file backed up: $BACKUP_FILE"
else
    log "⚠ Warning: original data file not found: $DATA_FILE"
fi


cp "$SCRIPT_DIR/run_simulation_round_generative_tiger.sh" "$STORAGE_DIR/"

cp "$SCRIPT_DIR/generative_recommend_tiger.sh" "$STORAGE_DIR/" 2>/dev/null || true
log "✓ Script has been backed up to $STORAGE_DIR"


if [ -d "$PROJECT_ROOT/recommenders/generativerec/ID_generation/preprocessing/processed/$DATASET_NAME" ]; then
    cp -r "$PROJECT_ROOT/recommenders/generativerec/ID_generation/preprocessing/processed/$DATASET_NAME" "$STORAGE_DIR/data"
    log "✓ Data directory backed up to $STORAGE_DIR/data/"
else
    error "Data directory does not exist: $PROJECT_ROOT/recommenders/generativerec/ID_generation/preprocessing/processed/$DATASET_NAME"
fi


cp "$DATA_FILE" "$STORAGE_DIR/data/$DATASET_NAME-1.txt"
log "✓ Initial data generated: $STORAGE_DIR/data/$DATASET_NAME-1.txt"


if [ "$TEST_MODE" == "True" ]; then
    SEMANTIC_ID_FILE="$PROJECT_ROOT/recommenders/generativerec/ID_generation/ID/${DATASET_NAME}_sentence-t5-xxl_42.pkl"
else
    SEMANTIC_ID_FILE="$PROJECT_ROOT/recommenders/generativerec/ID_generation/ID/${DATASET_NAME}_SASRec_42.pkl"
fi
if [ ! -f "$SEMANTIC_ID_FILE" ]; then
    log "Semantic ID file does not exist: $SEMANTIC_ID_FILE，please run RQ-VAE training first to generate codebook"
else
    log "✓ Semantic ID file exists: $SEMANTIC_ID_FILE"

    cp "$SEMANTIC_ID_FILE" "$STORAGE_DIR/data/"
    log "✓ Semantic ID file copied to: $STORAGE_DIR/data/"
fi
log ""
# ============================================

# ============================================
log "Step 3: Run simulation loop"
log "Switching to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"


CMD="uv run simulation/closed_loop_recommendation_system.py \
    --dataset $DATASET_NAME \
    --num_users $NUM_USERS \
    --backbone $BACKBONE \
    --items_per_cycle $ITEMS_PER_CYCLE \
    --max_cycles $MAX_CYCLES \
    --start_cycle $START_CYCLE \
    --storage_dir $STORAGE_DIR \
    --recommendation_file $RECOMMENDATION_FILE \
    --recommend_script $PROJECT_ROOT/simulation/generative_recommend_tiger.sh \
    --max_concurrent_users $MAX_CONCURRENT_USERS"

[[ "$ENABLE_TRAINING" == "false" ]] && CMD="$CMD --no_training"

log "Executing command: $CMD"
log ""

START_TIME=$(date +%s)
if $CMD; then
    DURATION=$(( $(date +%s) - START_TIME ))
    
    log ""
    log "============================================================================"
    success "Simulation completed (total elapsed: ${DURATION}s)"
    log "============================================================================"
    

    NUM_JSON=$(find "$STORAGE_DIR" -name "*.json" 2>/dev/null | wc -l)
    log "Result summary:"
    log "  - Number of generated JSON files: $NUM_JSON"
    if [[ -f "$DATA_FILE" ]] && [[ -f "$BACKUP_FILE" ]]; then
        NEW_LINES=$(($(wc -l < "$DATA_FILE") - $(wc -l < "$BACKUP_FILE")))
        log "  - New data lines: $NEW_LINES"
    fi
    log "  - Storage directory: $STORAGE_DIR"
    log "  - Log file: $LOG_FILE"
    log "============================================================================"

else
    log ""
    log "============================================================================"
    error "Simulation run failed"
    log "============================================================================"
    

    if [[ -f "$BACKUP_FILE" ]]; then
        cp "$BACKUP_FILE" "$DATA_FILE"
        log "✓ Data restored to backup state"
    fi
    exit 1
fi

log ""
log "Script execution completed"
exit 0