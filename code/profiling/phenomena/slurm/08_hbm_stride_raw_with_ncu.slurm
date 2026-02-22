#!/bin/bash
#SBATCH --account=slai
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=code/profiling/phenomena-08-hbmstride-out.%j
#SBATCH --error=code/profiling/phenomena-08-hbmstride-err.%j
#SBATCH --time=1:30:00
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
BUILD_DIR="${SCRIPT_DIR}/build"
RESULT_DIR="${SCRIPT_DIR}/phenomena/results"
mkdir -p "${RESULT_DIR}"

RAW_CSV="${RESULT_DIR}/08_hbm_stride_raw.${SLURM_JOB_ID}.csv"
NCU_SUMMARY_CSV="${RESULT_DIR}/08_hbm_stride_ncu_summary.${SLURM_JOB_ID}.csv"

# Tunables via environment variables.
RAW_RUNS="${RAW_RUNS:-20}"
RAW_WARMUP="${RAW_WARMUP:-5}"
RAW_LAUNCHES_PER_RUN="${RAW_LAUNCHES_PER_RUN:-64}"
NCU_WARMUP="${NCU_WARMUP:-5}"
THREADS_PER_BLOCK="${THREADS_PER_BLOCK:-256}"
BLOCKS_PER_SM="${BLOCKS_PER_SM:-4}"
N_POW2="${N_POW2:-30}"
PEAK_HBM_GBPS="${PEAK_HBM_GBPS:-1555}"

module load Stages/2025 CUDA/12 GCC/13.3.0

if ! command -v cmake >/dev/null 2>&1; then
  if ! module load CMake >/dev/null 2>&1; then
    CMAKE_CANDIDATE="$(module -t avail CMake 2>&1 | awk '/^CMake\// {print $1}' | tail -n1)"
    if [[ -n "${CMAKE_CANDIDATE}" ]]; then
      module load "${CMAKE_CANDIDATE}" >/dev/null 2>&1 || true
    fi
  fi
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found after module load. Try: module spider CMake"
  exit 1
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_CUDA_LINEINFO=ON \
  -DFORCE_O3=ON

cmake --build "${BUILD_DIR}" -j --target phenomenon_hbm_stride_interval_raw

echo "=== Raw stride sweep (requested-byte bandwidth) ==="
"${BUILD_DIR}/phenomenon_hbm_stride_interval_raw" \
  "${RAW_CSV}" \
  "${RAW_RUNS}" "${RAW_WARMUP}" "${RAW_LAUNCHES_PER_RUN}" \
  "${THREADS_PER_BLOCK}" "${BLOCKS_PER_SM}" "${N_POW2}" 0

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

DRAM_METRIC="$(pick_metric \
  "dram__throughput.avg.pct_of_peak_sustained_elapsed" \
  "dram__throughput.avg.pct_of_peak_sustained_active")"

if [[ -z "${DRAM_METRIC:-}" ]]; then
  echo "Could not find DRAM throughput metric in ncu --query-metrics output."
  exit 1
fi

STRIDES=(8 16 32 64 128 256 512 1024 2048 4096 8192)

echo "stride_bytes,dram_metric,dram_throughput_pct,dram_bandwidth_gbps,ncu_log_csv" > "${NCU_SUMMARY_CSV}"

echo "=== Nsight Compute per-stride DRAM throughput ==="
for STRIDE in "${STRIDES[@]}"; do
  NCU_LOG="${RESULT_DIR}/08_hbm_stride_ncu_stride_${STRIDE}.${SLURM_JOB_ID}.csv"
  DUMMY_CSV="${RESULT_DIR}/08_hbm_stride_ncu_dummy_${STRIDE}.${SLURM_JOB_ID}.csv"

  ncu --metrics "${DRAM_METRIC}" \
    -k hbm_stride_single_read_kernel \
    --launch-skip "${NCU_WARMUP}" --launch-count 1 \
    --csv --log-file "${NCU_LOG}" \
    "${BUILD_DIR}/phenomenon_hbm_stride_interval_raw" \
    "${DUMMY_CSV}" \
    1 "${NCU_WARMUP}" 1 "${THREADS_PER_BLOCK}" "${BLOCKS_PER_SM}" "${N_POW2}" "${STRIDE}"

  DRAM_PCT="$(
    python3 "${REPO_ROOT}/scripts/extract_ncu_metric_value.py" \
      --input "${NCU_LOG}" \
      --metric "${DRAM_METRIC}" \
      --kernel-substr "hbm_stride_single_read_kernel"
  )"

  DRAM_GBPS="$(awk -v pct="${DRAM_PCT}" -v peak="${PEAK_HBM_GBPS}" 'BEGIN { printf "%.8f", (pct * peak / 100.0) }')"

  echo "${STRIDE},${DRAM_METRIC},${DRAM_PCT},${DRAM_GBPS},${NCU_LOG}" >> "${NCU_SUMMARY_CSV}"
done

echo "Generated:"
echo "  ${RAW_CSV}"
echo "  ${NCU_SUMMARY_CSV}"
echo "Suggested plot command:"
echo "  python3 ${REPO_ROOT}/scripts/plot_hbm_stride_interval_raw.py --input ${RAW_CSV} --ncu-summary ${NCU_SUMMARY_CSV} --output ${REPO_ROOT}/images/hbm_stride_interval_raw.${SLURM_JOB_ID}.pdf"
