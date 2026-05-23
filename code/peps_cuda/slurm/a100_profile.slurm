#!/bin/bash -x
#SBATCH --account=<budget>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=peps-cuda-a100.%j.out
#SBATCH --error=peps-cuda-a100.%j.err

set -euo pipefail

module load CMake CUDA GCC || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${REPO_ROOT}/code/peps_cuda/build-a100"

"${REPO_ROOT}/code/peps_cuda/tools/check_cuda_env.sh"

cmake -S "${REPO_ROOT}/code/peps_cuda" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPEPS_CUDA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "${BUILD_DIR}" -j
ctest --test-dir "${BUILD_DIR}" --output-on-failure

NSYS_OUT="${REPO_ROOT}/code/peps_cuda/peps_cuda_a100_${SLURM_JOB_ID}"
NCU_OUT="${REPO_ROOT}/code/peps_cuda/peps_cuda_a100_ncu_${SLURM_JOB_ID}.csv"

srun --ntasks=1 --gpus-per-task=1 --cpus-per-task="${SRUN_CPUS_PER_TASK}" \
  nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  -o "${NSYS_OUT}" \
  "${BUILD_DIR}/peps_cuda_gpu_smoke"

srun --ntasks=1 --gpus-per-task=1 --cpus-per-task="${SRUN_CPUS_PER_TASK}" \
  ncu \
  --set=basic \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active \
  --csv --log-file "${NCU_OUT}" \
  "${BUILD_DIR}/peps_cuda_gpu_smoke"
