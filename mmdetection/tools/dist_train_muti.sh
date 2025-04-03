#!/usr/bin/env bash

CONFIG=$1
GPU_IDS=${2:-"0,1,2,3"}
NUM_RUNS=${3:-5}  # 要运行的总轮数，每轮4个任务

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

for RUN_ID in $(seq 1 $NUM_RUNS); do
  echo "========== Starting Experiment Group $RUN_ID =========="

  for ((i = 0; i < NUM_GPUS; i++)); do
    GPU_ID=${GPU_ARRAY[$i]}
    echo "  Launching task on GPU $GPU_ID (Experiment $RUN_ID)"

    OUTPUT_DIR="work_dirs/exp${RUN_ID}_gpu${GPU_ID}"
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="${OUTPUT_DIR}/train.log"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python $(dirname "$0")/train.py $CONFIG \
      --work-dir "$OUTPUT_DIR" \
      --launcher none "${@:4}" \
      > "$LOG_FILE" 2>&1 &

  done

  # 等待这一组全部跑完
  wait
  echo "========== Finished Experiment Group $RUN_ID =========="
done
