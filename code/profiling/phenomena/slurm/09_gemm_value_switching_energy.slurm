#!/bin/bash
#SBATCH --account=slai
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=code/profiling/phenomena-09-gemm-switching-out.%j
#SBATCH --error=code/profiling/phenomena-09-gemm-switching-err.%j
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

MERGED_CSV="${RESULT_DIR}/09_gemm_value_switching_raw.csv"

M="${M:-4096}"
N="${N:-4096}"
K="${K:-4096}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-1000}"
REPEATS="${REPEATS:-1}"
POWER_SAMPLE_MS="${POWER_SAMPLE_MS:-100}"

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

cmake --build "${BUILD_DIR}" -j --target phenomenon_gemm_value_switching

MODES=(all_zero zero_every_2 zero_every_3 zero_every_4 zero_every_5 normal uniform)

rm -f "${MERGED_CSV}"
first_mode=1
power_logger_pid=""

stop_power_logger() {
  if [[ -n "${power_logger_pid}" ]]; then
    kill "${power_logger_pid}" >/dev/null 2>&1 || true
    wait "${power_logger_pid}" 2>/dev/null || true
    power_logger_pid=""
  fi
}

trap stop_power_logger EXIT

for mode in "${MODES[@]}"; do
  MODE_CSV="${RESULT_DIR}/09_gemm_value_switching_${mode}.csv"
  POWER_CSV="${RESULT_DIR}/09_gemm_value_switching_power_${mode}.csv"

  echo "=== Mode: ${mode} ==="
  rm -f "${MODE_CSV}" "${POWER_CSV}"

  nvidia-smi \
    --query-gpu=timestamp,index,power.draw,utilization.gpu,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    -lms "${POWER_SAMPLE_MS}" > "${POWER_CSV}" &
  power_logger_pid="$!"

  sleep 1
  "${BUILD_DIR}/phenomenon_gemm_value_switching" \
    "${MODE_CSV}" \
    "${M}" "${N}" "${K}" \
    "${WARMUP}" "${ITERS}" "${REPEATS}" \
    "${mode}"
  sleep 1
  stop_power_logger

  if [[ "${first_mode}" -eq 1 ]]; then
    cat "${MODE_CSV}" > "${MERGED_CSV}"
    first_mode=0
  else
    tail -n +2 "${MODE_CSV}" >> "${MERGED_CSV}"
  fi
done

echo "Generated:"
echo "  ${MERGED_CSV}"
for mode in "${MODES[@]}"; do
  echo "  ${RESULT_DIR}/09_gemm_value_switching_${mode}.csv"
  echo "  ${RESULT_DIR}/09_gemm_value_switching_power_${mode}.csv"
done

