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
  -o "benchmark_scripts/noeager_qwen3_14b_tp2_bs128_out500_longbench_ctx14848_profile" \
  --force-overwrite=true \
  ./benchmark_scripts/run_noeager_qwen3_14b_tp2_bs128_out500_longbench_ctx14848.sh > "benchmark_scripts/noeager_qwen3_14b_tp2_bs128_out500_longbench_ctx14848_log.txt"
