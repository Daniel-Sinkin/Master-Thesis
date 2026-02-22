#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --disable-dcgm

set -euo pipefail

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}"
    exit 1
  fi
}

decode_b64() {
  local payload="$1"
  if [[ -z "${payload}" ]]; then
    printf ''
    return 0
  fi
  printf '%s' "${payload}" | base64 --decode
}

require_var REPO_ROOT
require_var EXPERIMENT_CU
require_var EXPERIMENT_NAME

if [[ ! -f "${EXPERIMENT_CU}" ]]; then
  echo "Experiment source not found: ${EXPERIMENT_CU}"
  exit 1
fi

RESULTS_BASE="${PROFILE_RESULTS_ROOT:-${REPO_ROOT}/code/profiling/results/experiments}"
JOB_DIR="${RESULTS_BASE}/job_${SLURM_JOB_ID}"
PREFIX="${EXPERIMENT_NAME}_"

mkdir -p "${JOB_DIR}"

exec > >(tee -a "${JOB_DIR}/${PREFIX}run.log")
exec 2>&1

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "EXPERIMENT_CU=${EXPERIMENT_CU}"
echo "JOB_DIR=${JOB_DIR}"
echo "Started (UTC): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

module load Stages/2025 CUDA/12 GCC/13.3.0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARCH="${CUDA_ARCH:-80}"
PROFILE_MODE="${PROFILE_MODE:-none}"
LIBS_CSV="${LIBS_CSV:-cudart}"
NSYS_TRACE="${NSYS_TRACE:-cuda,osrt}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"

RUN_ARGS_RAW="$(decode_b64 "${RUN_ARGS_B64:-}")"
EXTRA_NVCC_FLAGS_RAW="$(decode_b64 "${EXTRA_NVCC_FLAGS_B64:-}")"
NCU_METRICS_CSV="$(decode_b64 "${NCU_METRICS_B64:-}")"

if [[ -z "${NCU_METRICS_CSV}" ]]; then
  NCU_METRICS_CSV="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.sum.per_cycle_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.pct"
fi

RUN_ARGS=()
if [[ -n "${RUN_ARGS_RAW}" ]]; then
  read -r -a RUN_ARGS <<< "${RUN_ARGS_RAW}"
fi

EXTRA_NVCC_FLAGS=()
if [[ -n "${EXTRA_NVCC_FLAGS_RAW}" ]]; then
  read -r -a EXTRA_NVCC_FLAGS <<< "${EXTRA_NVCC_FLAGS_RAW}"
fi

NVCC_FLAGS=(-std=c++17 -O3 -lineinfo "-arch=sm_${ARCH}")
if [[ ${#EXTRA_NVCC_FLAGS[@]} -gt 0 ]]; then
  NVCC_FLAGS+=("${EXTRA_NVCC_FLAGS[@]}")
fi

LIB_FLAGS=()
IFS=',' read -r -a LIBS <<< "${LIBS_CSV}"
for lib in "${LIBS[@]}"; do
  case "${lib}" in
    "" ) ;;
    cudart) LIB_FLAGS+=("-lcudart") ;;
    cublas) LIB_FLAGS+=("-lcublas") ;;
    cublasLt|cublaslt) LIB_FLAGS+=("-lcublasLt") ;;
    curand) LIB_FLAGS+=("-lcurand") ;;
    cusolver) LIB_FLAGS+=("-lcusolver") ;;
    cusparse) LIB_FLAGS+=("-lcusparse") ;;
    *) LIB_FLAGS+=("-l${lib}") ;;
  esac
done

BIN_PATH="${JOB_DIR}/${PREFIX}bin"

echo "Compile command: nvcc ${EXPERIMENT_CU} ${NVCC_FLAGS[*]} ${LIB_FLAGS[*]} -o ${BIN_PATH}"
nvcc "${EXPERIMENT_CU}" "${NVCC_FLAGS[@]}" "${LIB_FLAGS[@]}" -o "${BIN_PATH}"

run_binary() {
  "${BIN_PATH}" "${RUN_ARGS[@]}"
}

run_nsys() {
  nsys profile \
    --force-overwrite true \
    --sample="${NSYS_SAMPLE}" \
    --trace="${NSYS_TRACE}" \
    --stats=true \
    -o "${JOB_DIR}/${PREFIX}nsys.${SLURM_JOB_ID}" \
    "${BIN_PATH}" "${RUN_ARGS[@]}"
}

run_ncu() {
  local out_csv="${JOB_DIR}/${PREFIX}ncu_metrics.${SLURM_JOB_ID}.csv"
  local cmd=(ncu --metrics "${NCU_METRICS_CSV}" --csv --log-file "${out_csv}")

  if [[ -n "${NCU_KERNEL_REGEX}" ]]; then
    cmd+=( -k "${NCU_KERNEL_REGEX}" )
  fi

  if [[ "${NCU_LAUNCH_SKIP}" != "" ]]; then
    cmd+=( --launch-skip "${NCU_LAUNCH_SKIP}" )
  fi

  if [[ "${NCU_LAUNCH_COUNT}" != "" ]]; then
    cmd+=( --launch-count "${NCU_LAUNCH_COUNT}" )
  fi

  cmd+=( "${BIN_PATH}" "${RUN_ARGS[@]}" )

  echo "NCU command: ${cmd[*]}"
  "${cmd[@]}"
}

echo "PROFILE_MODE=${PROFILE_MODE}"
echo "RUN_ARGS=${RUN_ARGS[*]}"

echo "=== Application output ===" > "${JOB_DIR}/${PREFIX}stdout.txt"

case "${PROFILE_MODE}" in
  none)
    run_binary | tee -a "${JOB_DIR}/${PREFIX}stdout.txt"
    ;;
  nsys)
    run_nsys
    ;;
  ncu)
    run_ncu
    ;;
  both)
    run_nsys
    run_ncu
    ;;
  *)
    echo "Unsupported PROFILE_MODE=${PROFILE_MODE}. Use one of: none, nsys, ncu, both"
    exit 1
    ;;
esac

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "experiment_name=${EXPERIMENT_NAME}"
  echo "experiment_cu=${EXPERIMENT_CU}"
  echo "profile_mode=${PROFILE_MODE}"
  echo "arch=${ARCH}"
  echo "libs_csv=${LIBS_CSV}"
  echo "run_args=${RUN_ARGS[*]}"
  echo "extra_nvcc_flags=${EXTRA_NVCC_FLAGS_RAW}"
  echo "nsys_trace=${NSYS_TRACE}"
  echo "nsys_sample=${NSYS_SAMPLE}"
  echo "ncu_metrics=${NCU_METRICS_CSV}"
  echo "ncu_kernel_regex=${NCU_KERNEL_REGEX}"
  echo "ncu_launch_skip=${NCU_LAUNCH_SKIP}"
  echo "ncu_launch_count=${NCU_LAUNCH_COUNT}"
  echo "hostname=$(hostname)"
  echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} > "${JOB_DIR}/${PREFIX}meta.txt"

TAR_PATH="${RESULTS_BASE}/${PREFIX}job_${SLURM_JOB_ID}.tar.gz"
tar -C "${RESULTS_BASE}" -czf "${TAR_PATH}" "job_${SLURM_JOB_ID}"
echo "${SLURM_JOB_ID}" > "${RESULTS_BASE}/${PREFIX}latest_job_id.txt"

echo "Generated outputs:"
echo "  ${JOB_DIR}/${PREFIX}run.log"
echo "  ${JOB_DIR}/${PREFIX}stdout.txt"
echo "  ${JOB_DIR}/${PREFIX}meta.txt"
echo "  ${TAR_PATH}"
echo "Done (UTC): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
