#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Navigate to the correct working directory
cd "/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench" || exit

# Nsys will execute the run script from the folder and drop logs right next to it
nsys profile \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --stats=true \
  -o "benchmark_scripts/qwen3_14b_pp2_bs256_out2000_longbench_ctx8192_profile" \
  --force-overwrite=true \
  ./benchmark_scripts/run_qwen3_14b_pp2_bs256_out2000_longbench_ctx8192.sh > "benchmark_scripts/qwen3_14b_pp2_bs256_out2000_longbench_ctx8192_log.txt"
