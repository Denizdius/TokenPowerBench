#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench" || exit

python3 run_single_node.py \
  --model "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3-14B" \
  --dataset longbench \
  --dataset-path "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench/longbench_en_nocode.jsonl" \
  --min-words 800 \
  --max-words 6000 \
  --batch-sizes 256 \
  --num-samples 1024 \
  --output-tokens 2000 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --max-model-len 14848 \
  --monitor gpu_only \
  --save-gpu-usage \
  --run-tag "qwen3_14b_dp2_bs256_out2000_longbench_ctx14848"

