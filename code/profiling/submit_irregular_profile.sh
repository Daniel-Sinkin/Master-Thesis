#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/results/irregular"
SLURM_SCRIPT="${SCRIPT_DIR}/nsight_irregular_training_profile.slurm"

mkdir -p "${RESULTS_BASE}"

job_id="$(sbatch --parsable \
  --chdir="${REPO_ROOT}" \
  --output="${RESULTS_BASE}/slurm-%j.out" \
  --error="${RESULTS_BASE}/slurm-%j.err" \
  --export=ALL,PROFILE_RESULTS_ROOT="${RESULTS_BASE}" \
  "${SLURM_SCRIPT}")"

echo "${job_id}" > "${RESULTS_BASE}/latest_job_id.txt"

echo "Submitted job ${job_id}"
echo "Track: squeue -j ${job_id} -o \"%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R\""
echo "Log:   ${RESULTS_BASE}/job_${job_id}/run.log"
echo "Tar:   ${RESULTS_BASE}/irregular_job_${job_id}.tar.gz"
echo "Local pull command:"
echo "  bash code/profiling/pull_irregular_results.sh <user@cluster-login> ${job_id} ${REPO_ROOT}"
