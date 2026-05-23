#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
project_dir="${repo_dir}/code/peps_cuda"

cmake -S "${project_dir}" -B "${project_dir}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${project_dir}/build"
ctest --test-dir "${project_dir}/build" --output-on-failure

cmake -S "${project_dir}" -B "${project_dir}/build-f32" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPEPS_CUDA_REAL_TYPE=float
cmake --build "${project_dir}/build-f32"
ctest --test-dir "${project_dir}/build-f32" --output-on-failure

python3 -m py_compile \
  "${project_dir}/julia_reference/validate_reference_fixtures.py" \
  "${project_dir}/julia_reference/summarize_reference_fixtures.py" \
  "${project_dir}/tools/benchmark_matrix.py" \
  "${project_dir}/tools/boundary_bucket_shapes.py" \
  "${project_dir}/tools/estimate_peps_costs.py" \
  "${project_dir}/tools/memory_pressure.py" \
  "${project_dir}/tools/occupancy_scratch.py"

python3 "${project_dir}/julia_reference/validate_reference_fixtures.py" \
  "${project_dir}/julia_reference/fixtures/reference_fixtures.jsonl"

python3 "${project_dir}/julia_reference/summarize_reference_fixtures.py" \
  "${project_dir}/julia_reference/fixtures/reference_fixtures.jsonl"

"${project_dir}/tools/check_cuda_env.sh"
