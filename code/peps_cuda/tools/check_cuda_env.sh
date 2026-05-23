#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname || true
date || true

echo
echo "== modules =="
if command -v module >/dev/null 2>&1; then
  module list 2>&1 || true
else
  echo "module command not available"
fi

echo
echo "== compilers =="
which cmake || true
cmake --version || true
which nvcc || true
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc not available"
fi
which g++ || true
g++ --version | head -n 1 || true

echo
echo "== nvidia =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
else
  echo "nvidia-smi not available"
fi

echo
echo "== slurm =="
env | grep -E '^(SLURM|CUDA_VISIBLE_DEVICES|OMP_NUM_THREADS|SRUN_CPUS_PER_TASK)=' | sort || true

echo
echo "== peps estimates =="
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/estimate_peps_costs.py" \
  --gpu jupiter_gh200 --lx 16 --ly 16 --d 8 --dc 64 --samples 2000 || true
python3 "${SCRIPT_DIR}/boundary_bucket_shapes.py" \
  --lx 16 --ly 16 --d 8 --dc 64 --dc-double 64 || true
