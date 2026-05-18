#!/bin/bash


mkdir -p res

# models=("GRU4Rec" "FMLPRecModel" "SASRec"  "LightSANs" "Mamba4Rec" "Linrec")
models=("SASRec")
datasets=("Toys_and_Games")
# "Sports_and_Outdoors" "Video_Games") # Office_Products


max_parallel=3
current_jobs=()
hidden_size=32

echo "Starting parallel experiments..."
echo "Model: ${models[*]}"
echo "Dataset: ${datasets[*]}"
echo "Max parallel jobs: $max_parallel"
echo "=========================================="


for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    echo "Starting task: $model on $dataset"
    

    uv run python -u run_finetune_full.py \
    --data_name="$dataset" \
    --ckp=0 \
    --hidden_size=$hidden_size \
    --backbone="$model" > "res/${dataset}-${model}-hidden_size=$hidden_size.txt" 2>&1 &
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
echo "All experiments completed!"
echo "Result files are saved in res/"
