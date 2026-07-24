#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench" || exit

python3 run_single_node.py \
  --model "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3.5-27B" \
  --dataset alpaca \
  --dataset-path "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench/alpaca.jsonl" \
  --batch-sizes 256 \
  --num-samples 1024 \
  --output-tokens 2000 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 2048 \
  --monitor gpu_only \
  --save-gpu-usage \
  --run-tag "qwen3.5_27b_single_bs256_out2000_noeager" \
  --language-model-only
