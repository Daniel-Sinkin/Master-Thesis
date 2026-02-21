#!/bin/bash
#SBATCH --account=slai
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=code/profiling/nsight-irregular-out.%j
#SBATCH --error=code/profiling/nsight-irregular-err.%j
#SBATCH --time=0:45:00
#SBATCH --disable-dcgm

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/code/profiling" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -d "/p/home/jusers/sinkin1/jureca/Master-Thesis/code/profiling" ]]; then
  REPO_ROOT="/p/home/jusers/sinkin1/jureca/Master-Thesis"
else
  echo "Could not locate repository root with code/profiling."
  exit 1
fi

SCRIPT_DIR="${REPO_ROOT}/code/profiling"
module load Stages/2025 CMake/3.29.3 CUDA/12 GCC/13.3.0
export CUDA_VISIBLE_DEVICES=0

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found after module load. Check CMake module availability."
  exit 1
fi

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_CUDA_LINEINFO=ON \
  -DFORCE_O3=ON
cmake --build "${SCRIPT_DIR}/build" -j

# Quick sanity runs
"${SCRIPT_DIR}/build/tn_irregular_direct_bad_seq" 64 47 84 64 1 1
"${SCRIPT_DIR}/build/tn_irregular_batched_good" 64 47 84 64 2 8

BAD_NSYS_ARGS=(128 47 84 64 1 4)
GOOD_NSYS_ARGS=(128 47 84 64 4 24)
BAD_NCU_ARGS=(128 47 84 64 1 4)
GOOD_NCU_ARGS=(128 47 84 64 10 80)

echo "=== Nsight Systems pass: BAD irregular pipeline ==="
nsys profile \
  --force-overwrite true \
  --sample=none \
  --trace=cuda,osrt \
  --stats=true \
  -o "${SCRIPT_DIR}/nsys_irregular_bad.${SLURM_JOB_ID}" \
  "${SCRIPT_DIR}/build/tn_irregular_direct_bad_seq" "${BAD_NSYS_ARGS[@]}"

echo "=== Nsight Systems pass: GOOD irregular pipeline ==="
nsys profile \
  --force-overwrite true \
  --sample=none \
  --trace=cuda,osrt \
  --stats=true \
  -o "${SCRIPT_DIR}/nsys_irregular_good.${SLURM_JOB_ID}" \
  "${SCRIPT_DIR}/build/tn_irregular_batched_good" "${GOOD_NSYS_ARGS[@]}"

for tag in bad good; do
  nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum \
    --format csv \
    --output "${SCRIPT_DIR}/nsys_irregular_${tag}_stats.${SLURM_JOB_ID}" \
    "${SCRIPT_DIR}/nsys_irregular_${tag}.${SLURM_JOB_ID}.nsys-rep" >/dev/null 2>&1 || true
done

ALL_METRICS="$(ncu --query-metrics --query-metrics-mode all 2>/dev/null || true)"

pick_metric() {
  for cand in "$@"; do
    if grep -Fq "${cand}" <<< "${ALL_METRICS}"; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

SELECTED_METRICS=()
MISSING_GROUPS=()

add_metric_group() {
  local label="$1"
  shift
  local picked
  if picked="$(pick_metric "$@")"; then
    SELECTED_METRICS+=("${picked}")
  else
    MISSING_GROUPS+=("${label}")
  fi
}

add_metric_group "sm_throughput" \
  "sm__throughput.avg.pct_of_peak_sustained_elapsed" \
  "sm__throughput.avg.pct_of_peak_sustained_active"
add_metric_group "dram_throughput" \
  "dram__throughput.avg.pct_of_peak_sustained_elapsed" \
  "dram__throughput.avg.pct_of_peak_sustained_active"
add_metric_group "l2_throughput" \
  "lts__t_sectors.avg.pct_of_peak_sustained_elapsed" \
  "lts__throughput.avg.pct_of_peak_sustained_elapsed" \
  "lts__throughput.avg.pct_of_peak_sustained_active"
add_metric_group "l1_hit_rate" \
  "l1tex__t_sector_hit_rate.pct"
add_metric_group "fp32_fma_pipe_active" \
  "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active" \
  "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"
add_metric_group "warps_active" \
  "sm__warps_active.avg.pct_of_peak_sustained_active" \
  "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
add_metric_group "occ_limit_registers" \
  "launch__occupancy_limit_registers"
add_metric_group "warps_eligible_per_cycle" \
  "smsp__warps_eligible.sum.per_cycle_active" \
  "smsp__warps_eligible.avg.per_cycle_active"
add_metric_group "issue_active" \
  "smsp__issue_active.avg.pct_of_peak_sustained_active" \
  "smsp__issue_active.avg.pct_of_peak_sustained_elapsed"
add_metric_group "thread_inst_per_inst" \
  "smsp__thread_inst_executed_per_inst_executed.pct" \
  "smsp__thread_inst_executed_per_inst_executed.ratio"

if [[ ${#SELECTED_METRICS[@]} -lt 6 ]]; then
  echo "Too few KPI metrics available (${#SELECTED_METRICS[@]} found)."
  echo "Discovered related metrics:"
  echo "${ALL_METRICS}" | grep -Ei "sm__throughput|dram__throughput|lts__|l1tex__|pipe_fma|warps_active|launch__occupancy_limit_registers|warps_eligible|issue_active|thread_inst_executed_per_inst_executed|pct_of_peak|repack|irregular" || true
  exit 1
fi

METRICS_CSV="$(IFS=,; echo "${SELECTED_METRICS[*]}")"

echo "Using ${#SELECTED_METRICS[@]} KPIs:"
printf '  - %s\n' "${SELECTED_METRICS[@]}"
if [[ ${#MISSING_GROUPS[@]} -gt 0 ]]; then
  echo "Missing KPI groups:"
  printf '  - %s\n' "${MISSING_GROUPS[@]}"
fi

echo "=== Nsight Compute pass: BAD repack kernel (memory/layout) ==="
ncu --metrics "${METRICS_CSV}" \
  -k tn_irregular_repack_bad_kernel \
  --launch-skip 1 --launch-count 1 \
  --csv --log-file "${SCRIPT_DIR}/bad_irregular_repack_metrics.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/tn_irregular_direct_bad_seq" "${BAD_NCU_ARGS[@]}"

echo "=== Nsight Compute pass: BAD direct kernel (serial launch regime) ==="
ncu --metrics "${METRICS_CSV}" \
  -k tn_irregular_direct_bad_kernel \
  --launch-skip 32 --launch-count 2 \
  --csv --log-file "${SCRIPT_DIR}/bad_irregular_direct_metrics.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/tn_irregular_direct_bad_seq" "${BAD_NCU_ARGS[@]}"

echo "=== Nsight Compute pass: GOOD batched kernel ==="
ncu --metrics "${METRICS_CSV}" \
  -k tn_irregular_batched_good_kernel \
  --launch-skip 20 --launch-count 2 \
  --csv --log-file "${SCRIPT_DIR}/good_irregular_batched_metrics.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/tn_irregular_batched_good" "${GOOD_NCU_ARGS[@]}"

echo "Generated outputs:"
echo "  ${SCRIPT_DIR}/nsys_irregular_bad.${SLURM_JOB_ID}.nsys-rep"
echo "  ${SCRIPT_DIR}/nsys_irregular_good.${SLURM_JOB_ID}.nsys-rep"
echo "  ${SCRIPT_DIR}/bad_irregular_repack_metrics.${SLURM_JOB_ID}.csv"
echo "  ${SCRIPT_DIR}/bad_irregular_direct_metrics.${SLURM_JOB_ID}.csv"
echo "  ${SCRIPT_DIR}/good_irregular_batched_metrics.${SLURM_JOB_ID}.csv"

