#!/bin/bash

# 1. Define absolute base paths
BASE_DIR="/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench"
MODEL_14B="/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3-14B"
MODEL_27B="/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3.5-27B"
DATASET_PATH="${BASE_DIR}/alpaca.jsonl"

# 2. Define Workloads (Batch_Output)
WORKLOADS=("128_500" "256_500" "256_2000")

# 3. Define Strategies
STRATS_14B=("single" "tp2" "dp2" "pp2")
# Added pp2_dp2 to complete the 4-GPU combinations
STRATS_27B=("single" "tp2" "dp2" "pp2" "tp4" "dp4" "pp4" "tp2_dp2" "tp2_pp2" "pp2_dp2")

# Create output directory to keep your workspace clean
SCRIPTS_DIR="benchmark_scripts"
mkdir -p "$SCRIPTS_DIR"

# 4. Generator Function
generate_scripts() {
    local model_name=$1
    local model_path=$2
    local strat=$3
    local bs=$4
    local out=$5

    local tp=1
    local pp=1
    local dp=1

    case $strat in
        "single") ;;
        "tp2") tp=2 ;;
        "dp2") dp=2 ;;
        "pp2") pp=2 ;;
        "tp4") tp=4 ;;
        "dp4") dp=4 ;;
        "pp4") pp=4 ;;
        "tp2_dp2") tp=2; dp=2 ;;
        "tp2_pp2") tp=2; pp=2 ;;
        "pp2_dp2") pp=2; dp=2 ;;
    esac

    # Add the specific flag ONLY if the model is Qwen3.5-27B
    local extra_flags=""
    local run_tag_suffix=""
    if [ "$model_name" == "qwen3.5_27b" ]; then
        extra_flags="  --language-model-only"
        run_tag_suffix=" \\"
    fi

    local base_name="${model_name}_${strat}_bs${bs}_out${out}"
    local run_script="${SCRIPTS_DIR}/run_${base_name}.sh"
    local nsys_script="${SCRIPTS_DIR}/nsys_${base_name}.sh"

    # --- Write the Execution Script ---
    cat <<EOT > "$run_script"
#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "${BASE_DIR}" || exit

python3 run_single_node.py \\
  --model "$model_path" \\
  --dataset alpaca \\
  --dataset-path "$DATASET_PATH" \\
  --batch-sizes $bs \\
  --num-samples 1024 \\
  --output-tokens $out \\
  --gpu-memory-utilization 0.90 \\
  --tensor-parallel-size $tp \\
  --pipeline-parallel-size $pp \\
  --data-parallel-size $dp \\
  --max-model-len 2048 \\
  --enforce-eager \\
  --monitor gpu_only \\
  --run-tag "$base_name"${run_tag_suffix}
${extra_flags}
EOT
    chmod +x "$run_script"

    # --- Write the Nsys Profiling Script ---
    cat <<EOT > "$nsys_script"
#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Navigate to the correct working directory
cd "${BASE_DIR}" || exit

# Nsys will execute the run script from the folder and drop logs right next to it
nsys profile \\
  --trace=cuda \\
  --sample=none \\
  --cpuctxsw=none \\
  --trace-fork-before-exec=true \\
  --stats=true \\
  -o "${SCRIPTS_DIR}/${base_name}_profile" \\
  --force-overwrite=true \\
  ./${SCRIPTS_DIR}/run_${base_name}.sh > "${SCRIPTS_DIR}/${base_name}_log.txt"
EOT
    chmod +x "$nsys_script"
}

# 5. Generate 14B Scripts
for w in "${WORKLOADS[@]}"; do
    bs=$(echo $w | cut -d'_' -f1)
    out=$(echo $w | cut -d'_' -f2)
    for s in "${STRATS_14B[@]}"; do
        generate_scripts "qwen3_14b" "$MODEL_14B" "$s" "$bs" "$out"
    done
done

# 6. Generate 27B Scripts
for w in "${WORKLOADS[@]}"; do
    bs=$(echo $w | cut -d'_' -f1)
    out=$(echo $w | cut -d'_' -f2)
    for s in "${STRATS_27B[@]}"; do
        generate_scripts "qwen3.5_27b" "$MODEL_27B" "$s" "$bs" "$out"
    done
done

echo "Success! Scripts generated with absolute paths and clean result filenames."
