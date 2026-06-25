#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench" || exit

python3 run_single_node.py \
  --model "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3.5-27B" \
  --dataset alpaca \
  --dataset-path "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench/alpaca.jsonl" \
  --batch-sizes 256 \
  --num-samples 1024 \
  --output-tokens 500 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --data-parallel-size 2 \
  --max-model-len 2048 \
  --enforce-eager \
  --monitor gpu_only \
  --run-tag "qwen3.5_27b_pp2_dp2_bs256_out500" \
  --language-model-only
