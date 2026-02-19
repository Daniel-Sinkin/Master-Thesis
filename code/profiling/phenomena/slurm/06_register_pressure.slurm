#!/bin/bash
#SBATCH --account=slai
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=code/profiling/phenomena-06-register-out.%j
#SBATCH --error=code/profiling/phenomena-06-register-err.%j
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
"${SCRIPT_DIR}/build/phenomenon_register_pressure" > "${RESULT_DIR}/06_register_pressure.${SLURM_JOB_ID}.csv"
