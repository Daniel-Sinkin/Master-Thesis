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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

if ! ncu --query-metrics | grep -Fq "${KPI}"; then
  echo "Metric not found: ${KPI}"
  echo "Candidate throughput metrics:"
  ncu --query-metrics | grep -Ei "sm__throughput.*pct_of_peak"
  exit 1
fi

ncu --metrics "${KPI}" \
  --launch-skip 20 --launch-count 1 \
  --csv --log-file "${SCRIPT_DIR}/good_fp32_tpp.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/gemm_fp32_good_cublas"

ncu --metrics "${KPI}" \
  -k naive_sgemm_fp32 \
  --launch-skip 5 --launch-count 1 \
  --csv --log-file "${SCRIPT_DIR}/bad_fp32_tpp.${SLURM_JOB_ID}.csv" \
  "${SCRIPT_DIR}/build/gemm_fp32_bad_naive"
