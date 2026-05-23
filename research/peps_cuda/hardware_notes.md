# Hardware Notes For A100, H100/H200, And JUPITER GH200

## Current Target Summary

| Target | GPU memory | HBM bandwidth | FP64 core peak | FP64 tensor peak | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| A100 SXM 40GB | 40 GB HBM2e | 1.555 TB/s | 9.7 TFLOP/s | 19.5 TFLOP/s | Development/profiling baseline. |
| H100 SXM 80GB | 80 GB HBM3 | 3.35 TB/s | 34 TFLOP/s | 67 TFLOP/s | Generic Hopper cloud target. |
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 34 TFLOP/s | 67 TFLOP/s | Useful cloud target, more memory/bandwidth. |
| JUPITER GH200 | 96 GB HBM3 | about 4 TB/s | H100-class | H100-class | Current documented JUPITER Booster target. |

Sources:
- Rechecked against public JUPITER/NVIDIA docs on 2026-05-15.
- JUPITER configuration docs list each Booster node as 4x GH200, each with a
  Hopper GPU, 132 SMs, 96 GB HBM3, and about 4 TB/s HBM bandwidth.
- JUPITER GPU docs currently warn that the system is in Early-Access and details
  may change, so these specs should be rechecked before a serious benchmark run.
- JUPITER technical overview says each Booster node has four GH200 superchips,
  288 Arm CPU cores total, and NVLink4 GPU-to-GPU connectivity.
- NVIDIA H200 docs list 141 GB HBM3e and 4.8 TB/s bandwidth for H200 SXM.
  This is a useful cloud target, but it is not what the current public JUPITER
  Booster configuration page lists.
- NVIDIA Hopper tuning docs provide the SM90 occupancy/shared-memory limits.
- Local datasheet copies are stored as `nvidia_a100_datasheet.pdf`,
  `nvidia_h100_datasheet.pdf`, and `nvidia_h200_datasheet.pdf`.

## JUPITER Node Implications

- Use one MPI rank per GPU first. JUPITER's Slurm setup maps one GPU to one task
  via `CUDA_VISIBLE_DEVICES` by default.
- A Booster node has four Grace CPUs and four Hopper GPUs. Each GPU has a nearby
  Grace CPU through NVLink-C2C, but the PEPS hot loop should still keep tensors,
  environments, and sample batches in HBM.
- CPU-GPU NVLink-C2C bandwidth is high enough that CPU orchestration and staging
  are less painful than PCIe, but not high enough to justify moving hot
  contraction operands in and out of HBM every sample.
- GPU-to-GPU links are useful for reduction/sharding later. The first scalable
  decomposition should shard samples across GPUs, because samples are naturally
  independent once a double-layer environment version exists.

## Hopper Kernel Constraints

From the Hopper tuning guide:

- 64 resident warps per SM.
- 64K 32-bit registers per SM.
- 255 registers/thread maximum.
- 32 thread blocks per SM maximum.
- 228 KB shared memory per SM.
- 227 KB shared memory per block with dynamic shared-memory opt-in.
- L2 cache is 50 MB on H100-class Hopper.
- TMA supports 1D-5D global/shared-memory tensor movement and can reduce register
  pressure for carefully designed Hopper-only kernels.

Practical interpretation:
- Do not chase maximum occupancy blindly. Boundary contractions need arithmetic
  intensity and reuse.
- Shared-memory-heavy custom kernels can look much better on H100/GH200 than
  A100, but if the A100 fallback is important, occupancy may drop earlier.
- Use cuBLASLt/grouped GEMM/cuTENSOR before writing SM90-only TMA kernels.

## Profiling Caveats

- JUPITER docs note that Nsight Compute may lock clocks to base frequency by
  default. This makes metric replay more deterministic but can make absolute
  timings differ from normal boosted runs.
- Record clock behavior, CUDA version, module stage, and `nvidia-smi` output with
  every serious benchmark.
- Run Nsight Systems before Nsight Compute. For this project, launch count and
  stream gaps can easily dominate before any one kernel deserves hand tuning.

## Memory Consequences For PEPS

Dense `O` is the immediate memory trap:

- `16x16, D=8, d=2, Ns=2000`: about 2.1M bulk parameters and 62.5 GiB dense
  complex-FP64 `O` storage.
- `32x32, D=8, d=2, Ns=5000`: about 8.4M bulk parameters and 625 GiB dense
  complex-FP64 `O` storage.
- Even on an H200 141 GB target, `32x32,D=8,Ns=5000` compact sampled-sector
  complex-FP32 `O` is about 156.5 GiB before workspaces, so extra HBM does not
  eliminate direct Gram/batching pressure.

Therefore:

- Dense `O` is a correctness/profiling baseline, not the production layout.
- The production path should either store only sampled physical sectors or
  accumulate `O O^dagger` directly by site/bucket.
- The `Ns x Ns` Gram is much smaller: about 61 MiB for `Ns=2000` and 381 MiB for
  `Ns=5000` in complex FP64.
