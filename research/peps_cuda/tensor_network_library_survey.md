# Tensor-Network / CUDA Library Survey

Purpose: identify libraries worth reusing or benchmarking before writing custom
kernels.

## NVIDIA cuTENSOR

Use:

- dense tensor contractions, reductions, permutations, and elementwise kernels,
- benchmarking contraction expressions that otherwise require explicit
  reshape/transpose plus cuBLAS.

Why useful:

- Official CUDA tensor primitive library.
- Gives optimized contraction kernels without building every layout path by
  hand.
- Especially relevant for boundary-MPS row absorption if reshape/transposes
  become expensive.

Limit:

- It is a primitive library, not a PEPS sampling algorithm.
- Need explicit descriptors and workspace management; still our responsibility
  to batch by shape and reuse buffers.

## NVIDIA cuTensorNet / cuQuantum

Use:

- contraction-path planning baseline,
- one-off exact/sliced tensor-network contractions,
- sanity checking small/medium contraction order decisions.

Why useful:

- Official NVIDIA tensor-network library with C/Python APIs and path-finding.
- Supports slicing and distributed contraction modes.

Limit:

- PEPS direct sampling repeatedly updates projected rows and environments; a
  generic path planner may not exploit the boundary-MPS reuse structure as well
  as a specialized implementation.
- The target algorithm is not arbitrary full-network contraction; it is a
  repeated boundary/environment workflow.

## cuBLAS / cuBLASLt / Grouped GEMM

Use:

- first production backend for row absorption and Gram formation.

Why useful:

- Most PEPS contractions should be lowered to GEMM buckets before custom kernel
  work.
- cuBLASLt heuristic tuning and grouped GEMM reduce launch overhead for many
  small/medium contractions.

Limit:

- Irregular ragged open-boundary shapes and many tiny buckets may still leave
  performance on the table.

## CUTLASS / CuTe / cuBLASDx

Use:

- second-pass custom kernels after profiler evidence.

Why useful:

- Lets us fuse small fixed-shape contractions, projections, scaling, and local
  reductions.
- Useful when grouped GEMM launch overhead or layout conversion dominates.
- cuBLASDx is a device-side BLAS extension, useful when a GEMM-like operation
  should live inside a larger CUDA kernel instead of becoming a separate library
  launch.
- CuTe/CUTLASS is the more general route for explicit layout algebra, tiled
  mainloops, and Hopper TMA/WGMMA-style kernels.

Limit:

- Higher engineering cost; not justified until cuBLAS/cuTENSOR baselines are
  measured.
- cuBLASDx is a preview/separate MathDx download, not a normal CUDA Toolkit
  dependency; check the installed module situation on A100/JUPITER before making
  it part of the baseline build.
- CuTe is powerful but easy to overuse. For PEPS, it should target only stable
  shape families from `boundary_bucket_shapes.py` or direct-Gram tiles, not the
  first correctness implementation.

## ITensors.jl / NDTensors

Use:

- semantic reference and fixture generation,
- not the production GPU backend.

Why useful:

- Excellent high-level tensor index model.
- The reference code already uses it and encodes the intended algorithm.

Limit:

- Current reference repo is not reproducible without manual package pins.
- The downloaded project does not configure CUDA.jl/GPU arrays.
- ITensor abstractions hide memory layout and batching decisions central to the
  thesis.

## Julia TensorOperations / CUDA.jl / cuTENSOR.jl

Use:

- possible exploratory prototypes for contraction expressions,
- not first C++ production target.

Why useful:

- Fast iteration on mathematical contractions.

Limit:

- Thesis implementation target is C++/CUDA for cluster deployment and profiler
  control.

## quimb / cotengra / opt_einsum

Use:

- contraction path sanity checks,
- literature comparison for hyper-optimized contraction order.

Limit:

- Python path optimizers do not replace boundary-MPS CUDA kernels and are not a
  natural production dependency for the supercomputer code.

## Practical Choice

Initial production stack:

```text
C++17 + CUDA
cuBLAS / cuBLASLt
cuSOLVER for compression/linear solves
cuTENSOR for selected dense tensor contractions
NCCL or MPI Allreduce for multi-GPU sample sharding
Nsight Systems / Nsight Compute / NVTX
```

Avoid custom tensor-core kernels until library baselines identify a fixed hot
shape family where they are justified.
