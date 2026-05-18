#!/bin/bash


if [ $# -lt 3 ]; then
    echo "Usage: $0 <models> <datasets> <cycle>"
    echo "Example: $0 'SASRec FMLPRecModel' 'Kindle_Store Toys_and_Games' 3"
    echo "      $0 GRU4Rec Kindle_Store 1"
    exit 1
fi


models_input="$1"
datasets_input="$2"
cycle="$3"


IFS=' ' read -ra models <<< "$models_input"
IFS=' ' read -ra datasets <<< "$datasets_input"


if ! [[ "$cycle" =~ ^[0-9]+$ ]]; then
    echo "Error: cycle argument must be numeric"
    exit 1
fi


mkdir -p "res/cycle/${dataset}"


max_parallel=2
current_jobs=()

echo "Starting parallel experiments..."
echo "Model: ${models[*]}"
echo "Dataset: ${datasets[*]}"
echo "Cycle: $cycle"
echo "Max parallel jobs: $max_parallel"
echo "=========================================="


for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    echo "Starting task: $model on $dataset"
    

    CUDA_VISIBLE_DEVICES=1 uv run python -u run_finetune_full_cycle.py \
    --data_name="$dataset" \
    --ckp=0 \
    --hidden_size=64 \
    --cycle="$cycle" \
    --backbone="$model" > "res/cycle/${dataset}/${model}-cycle${cycle}.txt" 2>&1 &
    

    job_pid=$!
    current_jobs+=($job_pid)
    echo "  Task PID: $job_pid"


    while [ ${#current_jobs[@]} -ge $max_parallel ]; do

      wait -n
      

      new_jobs=()
      for pid in "${current_jobs[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          new_jobs+=($pid)
        else
          echo "  Task $pid completed"
        fi
      done
      current_jobs=("${new_jobs[@]}")
    done
  done
done

echo "=========================================="
echo "Waiting for all remaining tasks to complete..."


for pid in "${current_jobs[@]}"; do
  echo "Waiting for task $pid to complete..."
  wait $pid
  echo "Task $pid completed"
done

echo "=========================================="
echo "All experiments completed！"
echo "Result files are saved in res/cycle"
