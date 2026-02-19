#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
MODE="${1:-afterok}"

if [[ "${MODE}" != "afterok" && "${MODE}" != "afterany" ]]; then
  echo "Usage: $0 [afterok|afterany]"
  exit 1
fi

scripts=(
  "00_build_phenomena.slurm"
  "01_divergence_sweep.slurm"
  "02_coalescing_sweep.slurm"
  "03_shared_bank_conflict.slurm"
  "04_launch_overhead.slurm"
  "05_ilp_dependency.slurm"
  "06_register_pressure.slurm"
  "07_arithmetic_intensity.slurm"
)

prev_job=""
for script in "${scripts[@]}"; do
  path="${SLURM_DIR}/${script}"
  if [[ ! -f "${path}" ]]; then
    echo "Missing script: ${path}"
    exit 1
  fi

  if [[ -z "${prev_job}" ]]; then
    job_id="$(sbatch --parsable "${path}")"
  else
    job_id="$(sbatch --parsable --dependency="${MODE}:${prev_job}" "${path}")"
  fi

  echo "${script} -> ${job_id}"
  prev_job="${job_id}"
done

echo "Queued chain tail job: ${prev_job}"
