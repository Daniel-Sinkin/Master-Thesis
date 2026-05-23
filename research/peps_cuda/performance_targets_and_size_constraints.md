# Performance Targets And Size Constraints

This is a planning target, not a promise. The point is to make thesis outcomes
measurable before the first cluster run.

## Memory Pressure

Current estimator:

```bash
python3 code/peps_cuda/tools/memory_pressure.py --lx 16 --ly 16 --d 8 --dc 64 --dc-double 16 --samples 2000 --hbm-gb 96
python3 code/peps_cuda/tools/memory_pressure.py --lx 32 --ly 32 --d 8 --dc 128 --dc-double 16 --samples 5000 --hbm-gb 96
```

Representative outputs:

- `16x16, D=8, Ns=2000`, complex FP64 dense `O`: `62.5 GiB`;
  compact sampled-sector `O`: `31.25 GiB`.
- The same case in complex FP32: dense `31.25 GiB`, compact `15.62 GiB`.
- `32x32, D=8, Ns=5000`, complex FP64 dense `O`: `625 GiB`;
  compact sampled-sector `O`: `312.5 GiB`.
- The same `32x32` case in complex FP32: dense `312.5 GiB`, compact
  `156.25 GiB`.

Conclusion: for JUPITER's documented 96 GB HBM per GH200 GPU, `32x32,D=8,Ns=5000`
cannot store dense or compact `O` on one GPU in FP64 or FP32. It needs direct
Gram accumulation, sample sharding across GPUs, smaller batches, checkpointed
`O`, or reduced precision/storage compression.

## Expected Handleable Inputs

CPU exact C++ oracle:

- `2x2`, `3x3`, small `D`, exact enumeration/debug only.

Single-GPU A100/H100/GH200 baseline:

- `8x8..16x16`, `D=4..8`, `Ns=1000..3000`, if direct/sampled-sector Gram avoids
  dense `O` or batches it.
- Rydberg/diagonal long-range terms are easier than non-diagonal long-range
  terms because diagonal energy does not require flipped contractions.
- Four-body local plaquette terms are plausible if bucketed and reused.

Four-GPU JUPITER node:

- Sample sharding should make `16x16,D=8,Ns=5000` plausible.
- `32x32,D=8` is plausible only with direct Gram accumulation and careful
  environment reuse; storing all rows is not plausible.

## Outcome Ladder

Bare minimum:

- C++ CPU oracle and CUDA smoke kernels run.
- Generated Julia fixtures can be exported and compared on tiny systems.
- A100 run produces Nsight traces showing where time goes.
- Memory estimator explains why dense `O` is not the production path.

Okay thesis:

- GPU code reproduces Julia tiny fixtures and runs `8x8,D<=4` end-to-end.
- Boundary-MPS contractions use cuBLAS/cuBLASLt buckets.
- Clear profiler-driven speedup over Julia reference for the same samples.

Good:

- `16x16,D=6..8,Ns~1000-2000` imaginary-time iteration runs on A100/H100-class
  hardware with sampled-sector/direct Gram.
- FP32 and FP64 pathways are compared with physics-facing error metrics.
- Bottlenecks are identified by Nsight Systems and Nsight Compute, not guessed.

Very good:

- JUPITER node-level sample sharding works across four GH200 GPUs.
- `16x16,D=8,Ns~5000` is practical with direct Gram or compact batching.
- The implementation has stable regression fixtures against Julia.

Above expectations:

- `32x32,D=8` becomes practical for a restricted Hamiltonian/imaginary-time
  benchmark by avoiding materialized `O` and using stale double-layer sampling
  efficiently.
- CUDA code is competitive with recent single-GPU PEPS-tVMC style work on a
  narrow measured slice.

Dream scenario:

- The project finds a real bottleneck breaker: for example a direct
  sampled-sector Gram/environment reuse strategy that makes a previously
  memory-prohibitive finite-PEPS regime usable on one JUPITER node.
- This is an outlook, not the planning baseline.

## Precision Policy

- FP64: correctness baseline and thesis reference.
- FP32/complex FP32: first performance target; halves `O`, Gram, and environment
  storage.
- TF32/tensor cores: benchmark for boundary GEMMs only after correctness
  fixtures exist.
- FP16/BF16/FP8 storage: future experiment for `O`/environment checkpoints, not
  first physics result.
- Ozaki/Ozaki-II: future backend for large compute-bound GEMM buckets or
  Blackwell-like weak-FP64 hardware; not first-pass Hopper work.
