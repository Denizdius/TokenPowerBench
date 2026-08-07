#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench" || exit

python3 run_single_node.py \
  --model "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3-14B" \
  --dataset alpaca \
  --dataset-path "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench/alpaca.jsonl" \
  --batch-sizes 256 \
  --num-samples 1024 \
  --output-tokens 2000 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 8192 \
  --enforce-eager \
  --monitor gpu_only \
  --run-tag "qwen3_14b_tp2_bs256_out2000_ctx8192"

