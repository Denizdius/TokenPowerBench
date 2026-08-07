#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench" || exit

python3 run_single_node.py \
  --model "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3-14B" \
  --dataset alpaca \
  --dataset-path "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench/alpaca.jsonl" \
  --batch-sizes 128 \
  --num-samples 1024 \
  --output-tokens 500 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --max-model-len 8192 \
  --monitor gpu_only \
  --run-tag "qwen3_14b_dp2_bs128_out500_noeager_ctx8192"

