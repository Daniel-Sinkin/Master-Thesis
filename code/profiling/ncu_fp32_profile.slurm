#!/bin/bash
#SBATCH --account=slai
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=code/profiling/ncu-fp32-out.%j
#SBATCH --error=code/profiling/ncu-fp32-err.%j
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
"${SCRIPT_DIR}/build/gemm_fp32_good_cublas"
"${SCRIPT_DIR}/build/gemm_fp32_bad_naive"

KPI="sm__throughput.avg.pct_of_peak_sustained_elapsed"

ALL_METRICS="$(ncu --query-metrics --query-metrics-mode all 2>/dev/null || true)"

if ! grep -Fq "${KPI}" <<< "${ALL_METRICS}"; then
  for cand in \
    "sm__throughput.avg.pct_of_peak_sustained_active" \
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active" \
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"; do
    if grep -Fq "${cand}" <<< "${ALL_METRICS}"; then
      KPI="${cand}"
      break
    fi
  done
fi

if ! grep -Fq "${KPI}" <<< "${ALL_METRICS}"; then
  echo "No expected FP32 throughput KPI found in this Nsight version."
  echo "Candidates discovered on this system:"
  echo "${ALL_METRICS}" | grep -Ei "sm__throughput|pipe_fma|flop_count_sp|pct_of_peak" || true
  exit 1
fi

echo "Using KPI: ${KPI}"

ncu --metrics "${KPI}" \
  --launch-skip 20 --launch-count 1 \
  --csv --log-file "${SCRIPT_DIR}/good_fp32_tpp.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/gemm_fp32_good_cublas"

ncu --metrics "${KPI}" \
  -k naive_sgemm_fp32 \
  --launch-skip 5 --launch-count 1 \
  --csv --log-file "${SCRIPT_DIR}/bad_fp32_tpp.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/gemm_fp32_bad_naive"
