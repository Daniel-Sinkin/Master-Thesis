#!/bin/bash
#SBATCH --account=slai
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=code/profiling/ncu-warpdiv-out.%j
#SBATCH --error=code/profiling/ncu-warpdiv-err.%j
#SBATCH --time=0:30:00
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

module load Stages/2025 CUDA/12 GCC/13.3.0
export CUDA_VISIBLE_DEVICES=0

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_CUDA_LINEINFO=ON \
  -DFORCE_O3=ON
cmake --build "${SCRIPT_DIR}/build" -j

# Quick sanity runs
"${SCRIPT_DIR}/build/warp_divergence_good_uniform"
"${SCRIPT_DIR}/build/warp_divergence_bad_divergent"

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
  echo "${ALL_METRICS}" | grep -Ei "sm__throughput|dram__throughput|lts__|l1tex__|pipe_fma|warps_active|launch__occupancy_limit_registers|warps_eligible|issue_active|thread_inst_executed_per_inst_executed|pct_of_peak|branch|uniform|diverg" || true
  exit 1
fi

METRICS_CSV="$(IFS=,; echo "${SELECTED_METRICS[*]}")"

echo "Using ${#SELECTED_METRICS[@]} KPIs:"
printf '  - %s\n' "${SELECTED_METRICS[@]}"
if [[ ${#MISSING_GROUPS[@]} -gt 0 ]]; then
  echo "Missing KPI groups:"
  printf '  - %s\n' "${MISSING_GROUPS[@]}"
fi

ncu --metrics "${METRICS_CSV}" \
  -k warp_uniform_branch \
  --launch-skip 20 --launch-count 1 \
  --csv --log-file "${SCRIPT_DIR}/warp_uniform_metrics.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/warp_divergence_good_uniform"

ncu --metrics "${METRICS_CSV}" \
  -k warp_divergent_branch \
  --launch-skip 20 --launch-count 1 \
  --csv --log-file "${SCRIPT_DIR}/warp_divergent_metrics.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/warp_divergence_bad_divergent"
