# Profiling KPIs For PEPS CUDA

This is the measurement checklist I would use for the first A100 and
JUPITER/GH200 profiling sessions. It is organized by question, because raw
Nsight metric dumps get unhelpful quickly.

## Always Record

- Git commit or diff bundle.
- CUDA toolkit and driver version.
- GPU model, memory size, clocks, and power mode.
- CMake flags, especially `CMAKE_CUDA_ARCHITECTURES`.
- Lattice/model: `Lx`, `Ly`, `D`, `Dc`, `d`, sample count.
- Precision mode and whether tensor cores are enabled.
- Whether profiler clock locking was active.

## NVTX Naming

Add NVTX ranges before serious cluster profiling. Suggested names:

```text
peps.step
peps.double_env.refresh
peps.sampling.batch
peps.sampling.row_conditional
peps.single_env.top_down
peps.horizontal_env.row
peps.energy.diagonal
peps.energy.horizontal
peps.energy.vertical
peps.energy.plaquette
peps.gradient.sampled_sector
peps.minsr.gram_dense
peps.minsr.gram_sampled_sector
peps.minsr.solve
peps.minsr.apply_odag
peps.minsr.apply_odag_sampled_sector
```

The goal is to make Nsight Systems answer "where did the iteration go?" without
manually mapping kernel names back to algorithm stages.

## Nsight Systems Questions

### Is the CPU launching too much?

Look for:

- Large gaps between CUDA kernels.
- Thousands of tiny kernels in one sample batch.
- Long CUDA API call bars from allocation, synchronization, or library setup.

Likely fixes:

- Grouped GEMM instead of per-site GEMM loops.
- CUDA Graph capture for stable repeated loops.
- Buffer pools and pre-created library handles/plans.
- Fusing cheap diagonal/one-site work into larger kernels.

### Are copies or migrations in the hot loop?

Look for:

- `cudaMemcpy` after initialization.
- Unified-memory page migrations.
- Host synchronization before every sample or row.

Likely fixes:

- Keep PEPS tensors, environments, samples, `E`, `O`, and Gram buffers resident
  on device.
- Move Hamiltonian flip records once per optimization step or when model data
  changes.
- Use pinned staging buffers only at batch boundaries.

### Is stream overlap real?

Look for:

- Double-layer environment refresh overlapping with sample/E/O work.
- Independent sample buckets running concurrently when kernels are small.
- Serialization caused by default-stream use or accidental synchronization.

Likely fixes:

- Separate streams for environment refresh, sampling, `E`, `O`, and minSR.
- Events for environment-version dependencies.
- Avoid `cudaDeviceSynchronize` outside timing/profiling boundaries.

## Nsight Compute Metrics By Bucket

### GEMM-Backed Boundary Absorption

Primary metrics:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- Tensor-core pipe utilization where applicable.
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`
- achieved occupancy and eligible warps.

Interpretation:

- Low SM and low DRAM: launch/library overhead or too little work per call.
- High DRAM, low SM: poor reuse or unnecessary transposes/copies.
- High SM, low tensor-core activity in low precision: layout/algorithm not
  using tensor-core-eligible paths.

### Custom Projection/Diagonal Kernels

Primary metrics:

- DRAM throughput.
- L2 hit rate.
- Branch/warp execution efficiency.
- Memory transactions per requested byte.

Interpretation:

- These kernels should be memory-light relative to contractions. If they show up
  high in the timeline, they need batching/fusion, not heroic math tuning.

### Sampled-Sector Gram

Primary metrics:

- DRAM throughput.
- L2 hit rate.
- SM throughput.
- Warp branch efficiency.
- Eligible warps per cycle.

Interpretation:

- The sampled-sector Gram has conditional site skips. If branch divergence is
  bad, bucket sample pairs by spin-agreement masks or accumulate by site rather
  than by pair.
- If DRAM dominates, avoid reading compact `O` rows multiple times by accumulating
  Gram tiles per site/block and using shared memory.

### Compression

Primary metrics:

- cuSOLVER call time.
- GEMM/eigensolver/SVD split.
- Workspace allocation time.
- Small-matrix batching efficiency.

Interpretation:

- If compression dominates, compare density-matrix compression, QR/SVD sweeps,
  and batched/spectrum-truncated variants before tuning row absorption further.

## Roofline Sanity

For every benchmark row, record:

```text
effective_flop_s = estimated_flops / elapsed_seconds
effective_byte_s = estimated_bytes / elapsed_seconds
arithmetic_intensity = estimated_flops / estimated_bytes
```

Use these only as triage. Boundary-MPS formulas are approximate and compression
changes real traffic, but the roofline view quickly separates:

- launch-bound tiny kernels,
- memory-bound layout/copy problems,
- compute-bound contraction kernels,
- solver/compression bottlenecks.

## First CSV Schema

For microbenchmarks, write one CSV row per run:

```text
timestamp,host,gpu,cuda,git,arch,lx,ly,D,Dc,d,Ns,
family,M,N,K,count,precision,backend,
elapsed_ms,estimated_flops,estimated_bytes,
effective_tflops,effective_gbs,notes
```

Suggested `backend` values:

```text
cpu_exact
cuda_custom
cublas_loop
cublas_strided_batched
cublas_grouped
cublaslt
cutensor
cusolver
cuda_graph
```

## Stop Conditions For An Optimization Pass

Stop tuning a kernel and move up a level when:

- It is below 5% of iteration time in Nsight Systems.
- It is already close to the relevant roofline and the remaining bottleneck is
  elsewhere.
- It is a debug-only dense path that production will replace.
- The algorithmic memory footprint is impossible at the target size.

For this project, the last condition matters a lot: dense `O` can be optimized
forever and still be the wrong production object for `32x32, D=8`.
