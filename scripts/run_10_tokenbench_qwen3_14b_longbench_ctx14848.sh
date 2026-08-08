#!/bin/bash
#SBATCH --job-name=tb_lb_eager_14b
#SBATCH --output=/gpfs/projects/etur83/tokenbench_qwen3_14b_longbench_multi_run_result/logs/master_longbench_eager_10run_%j.log
#SBATCH --error=/gpfs/projects/etur83/tokenbench_qwen3_14b_longbench_multi_run_result/logs/master_longbench_eager_10run_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --account=etur83
#
# 10× multi-run: Qwen3-14B LongBench EAGER (ctx14848, words 800–6000).
# Only globs LongBench scripts — never Alpaca *_ctx8192.sh.
#
# Before submit:
#   1) Sync TokenPowerBench + longbench_en_nocode.jsonl
#   2) Set TOKENBENCH_SIF / TOKENBENCH_BIND to match Alpaca 10×, OR edit below
#   3) Copy this file to /gpfs/projects/etur83/ (same place as Alpaca launchers)
#   sbatch run_10_tokenbench_qwen3_14b_longbench_ctx14848.sh
#
set -euo pipefail

MODE="eager"
SCRIPT_GLOB="run_qwen3_14b_*_longbench_ctx14848.sh"
OUT_ROOT="/gpfs/projects/etur83/tokenbench_qwen3_14b_longbench_multi_run_result"
REPO="/gpfs/projects/etur83/my_volume/lm-evaluation-harness/TokenPowerBench"
SCRIPTS_DIR="${REPO}/benchmark_scripts"
RESULTS_DIR="${REPO}/results"
NUM_RUNS=10

module load SINGULARITY/4.1.5 2>/dev/null || true
# Override these to match /gpfs/projects/etur83/run_10_tokenbench_qwen3_14b_ctx8192.sh
IMAGE="${TOKENBENCH_SIF:-/gpfs/projects/etur83/my_volume/containers/vllm.sif}"
BIND="${TOKENBENCH_BIND:-/gpfs/projects/etur83:/gpfs/projects/etur83}"

mkdir -p "${OUT_ROOT}/logs"
mapfile -t SCRIPTS < <(cd "${SCRIPTS_DIR}" && ls ${SCRIPT_GLOB} 2>/dev/null | sort)
if [ "${#SCRIPTS[@]}" -eq 0 ]; then
  echo "ERROR: no scripts matched ${SCRIPTS_DIR}/${SCRIPT_GLOB}"
  exit 1
fi

echo "=================================================="
echo "LongBench ${MODE} 10-run started at $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Output: ${OUT_ROOT}"
echo "Scripts per iteration: ${#SCRIPTS[@]}"
echo "Image: ${IMAGE}"
echo "=================================================="
for s in "${SCRIPTS[@]}"; do echo "  - ${s%.sh}"; done
echo "=================================================="

for run_i in $(seq 1 "${NUM_RUNS}"); do
  RUN_DIR="${OUT_ROOT}/run_${run_i}"
  JSON_DIR="${RUN_DIR}/jsons"
  LOG_DIR="${RUN_DIR}/qwen3_14b_tokenbench_result"
  mkdir -p "${JSON_DIR}" "${LOG_DIR}"

  echo ""
  echo "##################################################"
  echo "               STARTING RUN ${run_i} OF ${NUM_RUNS}"
  echo "##################################################"

  for script in "${SCRIPTS[@]}"; do
    name="${script%.sh}"
    log="${LOG_DIR}/logs_${name}_job_${SLURM_JOB_ID:-local}.txt"
    echo "--------------------------------------------------"
    echo ">>> NOW RUNNING: ${name} (Run ${run_i})"
    echo ">>> LOG: ${log}"
    echo "--------------------------------------------------"

    marker="${RESULTS_DIR}/.collect_marker_$$"
    touch "${marker}"
    set +e
    singularity exec --nv --bind "${BIND}" "${IMAGE}" \
      bash "${SCRIPTS_DIR}/${script}" > "${log}" 2>&1
    rc=$?
    set -e
    if [ "${rc}" -ne 0 ]; then
      echo "WARNING: ${name} failed rc=${rc} (run ${run_i}); continuing"
    fi

    count=0
    while IFS= read -r f; do
      [ -z "${f}" ] && continue
      base=$(basename "${f}")
      # Eager: keep *longbench_ctx14848* but not *noeager*
      if [[ "${base}" == *longbench_ctx14848*.json && "${base}" != *noeager* ]]; then
        cp -f "${f}" "${JSON_DIR}/"
        count=$((count + 1))
      fi
    done < <(find "${RESULTS_DIR}" -maxdepth 1 -type f -name '*.json' -newer "${marker}" 2>/dev/null | sort)
    rm -f "${marker}"
    echo ">>> Collected ${count} JSON(s) into ${JSON_DIR}"
  done
done

echo "=================================================="
echo "LongBench ${MODE} 10-run finished at $(date)"
echo "=================================================="
