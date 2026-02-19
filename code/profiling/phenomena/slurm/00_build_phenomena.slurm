#!/bin/bash
#SBATCH --account=slai
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=code/profiling/phenomena-00-build-out.%j
#SBATCH --error=code/profiling/phenomena-00-build-err.%j
#SBATCH --time=0:20:00
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
RESULT_DIR="${SCRIPT_DIR}/phenomena/results"
mkdir -p "${RESULT_DIR}"

module load Stages/2025 CUDA/12 GCC/13.3.0

if ! command -v cmake >/dev/null 2>&1; then
  if ! module load CMake >/dev/null 2>&1; then
    CMAKE_CANDIDATE="$(module -t avail CMake 2>&1 | awk '/^CMake\\// {print $1}' | tail -n1)"
    if [[ -n "${CMAKE_CANDIDATE}" ]]; then
      module load "${CMAKE_CANDIDATE}" >/dev/null 2>&1 || true
    fi
  fi
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found after module load. Try: module spider CMake"
  exit 1
fi

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_CUDA_LINEINFO=ON \
  -DFORCE_O3=ON

cmake --build "${SCRIPT_DIR}/build" -j

echo "Build complete. Binaries in ${SCRIPT_DIR}/build"
