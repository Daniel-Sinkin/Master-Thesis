#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 <remote_login> [job_id|latest] [remote_repo_root] [local_dest]

Examples:
  $0 sinkin1@jrc0225 14507928
  $0 sinkin1@jrc0225 latest /p/home/jusers/sinkin1/jureca/Master-Thesis
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

REMOTE_LOGIN="$1"
JOB_ID="${2:-latest}"
REMOTE_REPO_ROOT="${3:-/p/home/jusers/sinkin1/jureca/Master-Thesis}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_DEST="${4:-${LOCAL_REPO_ROOT}/code/profiling/results/irregular}"

REMOTE_RESULTS_BASE="${REMOTE_REPO_ROOT}/code/profiling/results/irregular"

if [[ "${JOB_ID}" == "latest" ]]; then
  JOB_ID="$(ssh "${REMOTE_LOGIN}" "cat '${REMOTE_RESULTS_BASE}/latest_job_id.txt'" | tr -d '[:space:]')"
  if [[ -z "${JOB_ID}" ]]; then
    echo "Could not resolve latest job id from ${REMOTE_RESULTS_BASE}/latest_job_id.txt"
    exit 1
  fi
fi

mkdir -p "${LOCAL_DEST}"

REMOTE_JOB_DIR="${REMOTE_RESULTS_BASE}/job_${JOB_ID}"
REMOTE_TAR="${REMOTE_RESULTS_BASE}/irregular_job_${JOB_ID}.tar.gz"

if command -v rsync >/dev/null 2>&1; then
  rsync -avz "${REMOTE_LOGIN}:${REMOTE_JOB_DIR}/" "${LOCAL_DEST}/job_${JOB_ID}/"
  rsync -avz "${REMOTE_LOGIN}:${REMOTE_TAR}" "${LOCAL_DEST}/" || true
else
  mkdir -p "${LOCAL_DEST}/job_${JOB_ID}"
  scp -r "${REMOTE_LOGIN}:${REMOTE_JOB_DIR}/." "${LOCAL_DEST}/job_${JOB_ID}/"
  scp "${REMOTE_LOGIN}:${REMOTE_TAR}" "${LOCAL_DEST}/" || true
fi

echo "Pulled run ${JOB_ID} into ${LOCAL_DEST}"
echo "Run dir: ${LOCAL_DEST}/job_${JOB_ID}"
echo "Tar:     ${LOCAL_DEST}/irregular_job_${JOB_ID}.tar.gz"
