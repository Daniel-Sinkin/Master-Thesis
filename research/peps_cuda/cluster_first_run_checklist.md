# First Cluster Run Checklist

This is the concrete first-run plan for A100 and JUPITER/GH200 access. It is
biased toward getting trustworthy profiler traces quickly, not toward proving
the final algorithm on day one.

## Before Submitting

- Run `code/peps_cuda/tools/check_cuda_env.sh` in an interactive allocation or
  at the top of a first batch job and keep the output with the profile bundle.
- Run `code/peps_cuda/tools/run_cpu_regression.sh` locally before copying the
  worktree to the cluster, so FP64/FP32 CPU tests and Julia fixtures are already
  known-good.
- Confirm compiler and CUDA module versions.
- Confirm `nvidia-smi -L` exposes the expected GPU count.
- Build one clean Release tree per architecture:
  - A100: `-DCMAKE_CUDA_ARCHITECTURES=80`
  - H100/GH200/H200: `-DCMAKE_CUDA_ARCHITECTURES=90`
- On JUPITER Booster, follow the current public docs: request
  `--partition=booster` and `--gpus=4` for a full node, with
  `--gpus-per-task=1` for one rank per GPU.
- Run `ctest --output-on-failure` before profiling.
- Save the exact git commit hash and CMake cache for each profile bundle.

## Smoke Tests

Run the CPU exact tests first:

```bash
ctest --test-dir build --output-on-failure
```

For local no-CUDA validation before the cluster run:

```bash
code/peps_cuda/tools/run_cpu_regression.sh
```

Then run the CUDA smoke executable:

```bash
./build/peps_cuda_gpu_smoke
```

The smoke executable currently checks:

- `|psi|^2` weights.
- Uniform and ragged physical-slice projection.
- Diagonal Heisenberg, one-site, and two-site energy kernels.
- Importance-weight normalization plumbing.
- Dense minSR Gram assembly on tiny matrices.
- Sampled-sector minSR Gram assembly.
- Sampled-sector `O^dagger x` scatter back into dense parameter space.

## First Nsight Systems Pass

Use Nsight Systems before Nsight Compute. The first question is whether the run
has avoidable CPU gaps or launch storms.

Capture:

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true --force-overwrite=true \
  -o nsys_peps_smoke ./build/peps_cuda_gpu_smoke
```

Record:

- Total CUDA kernel launch count.
- Time spent in host-side CUDA API calls.
- Any host-device copies after initialization.
- CPU gaps between kernels.
- Whether streams overlap once the real sampler is wired.

## First Nsight Compute Pass

Only profile one kernel family at a time. For the initial smoke executable, the
dense Gram and ragged projection kernels are the useful early targets:

```bash
ncu --set full --target-processes all --force-overwrite \
  -o ncu_peps_smoke ./build/peps_cuda_gpu_smoke
```

Primary metrics to inspect:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`
- Branch efficiency for Hamiltonian bucket kernels.

## Benchmark Matrix

Generate a triage CSV before deciding which cases to run:

```bash
python3 code/peps_cuda/tools/benchmark_matrix.py --gpu jupiter_gh200 \
  --lattices 4x4,8x8,16x16,32x32 \
  --d-values 2,4,6,8 \
  --dc-values 16,32,64,96 \
  --samples 128,512,2000,5000
```

Interpretation:

- `dense-o-ok`: materializing dense `O` is likely acceptable for a first debug
  run.
- `sampled-sector-o-ok`: materialize only the sampled physical sector.
- `direct-gram-required`: skip materialized `O`; accumulate sample-space Gram
  directly from site slices or boundary environments.

Generate approximate boundary-MPS GEMM bucket shapes:

```bash
python3 code/peps_cuda/tools/boundary_bucket_shapes.py \
  --lx 16 --ly 16 --d 8 --dc 64 --dc-double 64
```

Use the shape rows as the first grouped-GEMM/cuTENSOR microbenchmark manifest.
They are approximate because real compression changes `chi` along the sweep, but
they are much closer to the PEPS workload than square GEMM benchmarks.

## A100 Order

1. `2x2` and `3x3` exact correctness.
2. CUDA smoke executable.
3. Synthetic dense minSR Gram with increasing `Ns`.
4. Synthetic ragged projection with open-boundary site shapes.
5. First boundary-MPS contraction bucket.
6. Direct-sampling row bucket.

## JUPITER/GH200 Order

1. Repeat the A100 smoke tests with SM90.
2. Confirm CPU/GPU affinity: one MPI rank per GH200 GPU and close Grace CPU.
3. Keep hot tensors in HBM; use Grace memory only for orchestration/staging.
4. Compare grouped GEMM, cuTENSOR, and custom kernels on the same synthetic
   boundary-MPS buckets.
5. Add CUDA Graph capture only once the launch topology is stable.

## What Would Count As A Bad Trace

- Repeated `cudaMalloc`/`cudaFree` in the sample loop.
- Thousands of sub-10-microsecond kernels for one sample batch.
- Host-device copies of PEPS tensors or environments after initialization.
- Low SM utilization with high DRAM pressure on contractions that should reuse
  data.
- Dense `O` allocation near HBM capacity, because profiler overhead and library
  workspaces will then make results noisy or fail outright.
