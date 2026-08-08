#!/bin/bash

# 1. Define absolute base paths
BASE_DIR="/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench"
MODEL_14B="/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3-14B"
MODEL_27B="/gpfs/projects/etur83/my_volume/lm-evaluation-harness/Qwen3.5-27B"
ALPACA_PATH="${BASE_DIR}/alpaca.jsonl"
LONGBENCH_PATH="${BASE_DIR}/longbench_en_nocode.jsonl"

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
# variant: "" | "noeager" | "noeager_ctx8192"
#         | "eager_longbench_ctx14848" | "noeager_longbench_ctx14848"
generate_scripts() {
    local model_name=$1
    local model_path=$2
    local strat=$3
    local bs=$4
    local out=$5
    local variant=${6:-}

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

    local gpu_util="0.90"
    local max_model_len=2048
    local eager_block="  --enforce-eager \\"
    local name_prefix=""
    local name_suffix=""
    local write_gpu_watch=0
    local dataset="alpaca"
    local dataset_path="$ALPACA_PATH"
    local word_filter_block=""
    local run_body_after_max_len="  --monitor gpu_only \\"
    local run_tag="${model_name}_${strat}_bs${bs}_out${out}"
    local artifact_prefix="${run_tag}"

    if [ "$variant" == "noeager" ]; then
        gpu_util="0.85"
        eager_block=""
        name_prefix="noeager_"
        write_gpu_watch=1
        run_tag="${run_tag}_noeager"
        artifact_prefix="noeager_${model_name}_${strat}_bs${bs}_out${out}"
    elif [ "$variant" == "noeager_ctx8192" ]; then
        gpu_util="0.90"
        max_model_len=8192
        eager_block=""
        name_prefix="noeager_"
        name_suffix="_ctx8192"
        write_gpu_watch=1
        run_tag="${run_tag}_noeager_ctx8192"
        artifact_prefix="noeager_${model_name}_${strat}_bs${bs}_out${out}_ctx8192"
    elif [ "$variant" == "eager_longbench_ctx14848" ]; then
        gpu_util="0.90"
        max_model_len=14848
        eager_block="  --enforce-eager \\"
        name_suffix="_longbench_ctx14848"
        write_gpu_watch=1
        dataset="longbench"
        dataset_path="$LONGBENCH_PATH"
        word_filter_block="  --min-words 800 \\
  --max-words 6000 \\"
        run_tag="${run_tag}_longbench_ctx14848"
        artifact_prefix="${model_name}_${strat}_bs${bs}_out${out}_longbench_ctx14848"
    elif [ "$variant" == "noeager_longbench_ctx14848" ]; then
        gpu_util="0.90"
        max_model_len=14848
        eager_block=""
        name_prefix="noeager_"
        name_suffix="_longbench_ctx14848"
        write_gpu_watch=1
        dataset="longbench"
        dataset_path="$LONGBENCH_PATH"
        word_filter_block="  --min-words 800 \\
  --max-words 6000 \\"
        run_tag="${run_tag}_noeager_longbench_ctx14848"
        artifact_prefix="noeager_${model_name}_${strat}_bs${bs}_out${out}_longbench_ctx14848"
    fi

    if [ -n "$eager_block" ]; then
        run_body_after_max_len="${eager_block}
  --monitor gpu_only \\"
    fi

    # Add the specific flag ONLY if the model is Qwen3.5-27B
    local extra_flags=""
    local run_tag_suffix=""
    if [ "$model_name" == "qwen3.5_27b" ]; then
        extra_flags="  --language-model-only"
        run_tag_suffix=" \\"
    fi

    local base_name="${model_name}_${strat}_bs${bs}_out${out}${name_suffix}"
    local run_script="${SCRIPTS_DIR}/run_${name_prefix}${base_name}.sh"
    local nsys_script="${SCRIPTS_DIR}/nsys_${name_prefix}${base_name}.sh"
    local gpu_watch_script="${SCRIPTS_DIR}/gpu_watch_${name_prefix}${base_name}.sh"

    # --- Write the Execution Script ---
    cat <<EOT > "$run_script"
#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "${BASE_DIR}" || exit

python3 run_single_node.py \\
  --model "$model_path" \\
  --dataset $dataset \\
  --dataset-path "$dataset_path" \\
${word_filter_block}
  --batch-sizes $bs \\
  --num-samples 1024 \\
  --output-tokens $out \\
  --gpu-memory-utilization $gpu_util \\
  --tensor-parallel-size $tp \\
  --pipeline-parallel-size $pp \\
  --data-parallel-size $dp \\
  --max-model-len $max_model_len \\
${run_body_after_max_len}
  --run-tag "$run_tag"${run_tag_suffix}
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
  -o "${SCRIPTS_DIR}/${artifact_prefix}_profile" \\
  --force-overwrite=true \\
  ./${SCRIPTS_DIR}/run_${name_prefix}${base_name}.sh > "${SCRIPTS_DIR}/${artifact_prefix}_log.txt"
EOT
    chmod +x "$nsys_script"

    if [ "$write_gpu_watch" -eq 1 ]; then
        cat <<EOT > "$gpu_watch_script"
#!/bin/bash

# Navigate to the correct working directory to find the python script
cd "${BASE_DIR}" || exit

python3 run_single_node.py \\
  --model "$model_path" \\
  --dataset $dataset \\
  --dataset-path "$dataset_path" \\
${word_filter_block}
  --batch-sizes $bs \\
  --num-samples 1024 \\
  --output-tokens $out \\
  --gpu-memory-utilization $gpu_util \\
  --tensor-parallel-size $tp \\
  --pipeline-parallel-size $pp \\
  --data-parallel-size $dp \\
  --max-model-len $max_model_len \\
${run_body_after_max_len}
  --save-gpu-usage \\
  --run-tag "$run_tag"${run_tag_suffix}
${extra_flags}
EOT
        chmod +x "$gpu_watch_script"
    fi
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

# 7. Generate no-eager variants (gpu util 0.85, no --enforce-eager)
for w in "${WORKLOADS[@]}"; do
    bs=$(echo $w | cut -d'_' -f1)
    out=$(echo $w | cut -d'_' -f2)
    for s in "${STRATS_14B[@]}"; do
        generate_scripts "qwen3_14b" "$MODEL_14B" "$s" "$bs" "$out" "noeager"
    done
done

for w in "${WORKLOADS[@]}"; do
    bs=$(echo $w | cut -d'_' -f1)
    out=$(echo $w | cut -d'_' -f2)
    for s in "${STRATS_27B[@]}"; do
        generate_scripts "qwen3.5_27b" "$MODEL_27B" "$s" "$bs" "$out" "noeager"
    done
done

# 8. Generate no-eager ctx8192 for Qwen3-14B (gpu util 0.90, no --enforce-eager)
for w in "${WORKLOADS[@]}"; do
    bs=$(echo $w | cut -d'_' -f1)
    out=$(echo $w | cut -d'_' -f2)
    for s in "${STRATS_14B[@]}"; do
        generate_scripts "qwen3_14b" "$MODEL_14B" "$s" "$bs" "$out" "noeager_ctx8192"
    done
done

# 9. LongBench ctx14848 for Qwen3-14B (eager + noeager; EN words 800–6000)
for w in "${WORKLOADS[@]}"; do
    bs=$(echo $w | cut -d'_' -f1)
    out=$(echo $w | cut -d'_' -f2)
    for s in "${STRATS_14B[@]}"; do
        generate_scripts "qwen3_14b" "$MODEL_14B" "$s" "$bs" "$out" "eager_longbench_ctx14848"
        generate_scripts "qwen3_14b" "$MODEL_14B" "$s" "$bs" "$out" "noeager_longbench_ctx14848"
    done
done

echo "Success! Scripts generated with absolute paths and clean result filenames."
