# Next Implementation Steps

This is the concrete continuation plan after the current scaffold.

## 1. Compile And Smoke On A100

Goal: prove the CUDA target builds and the smoke kernels behave on real hardware.

Commands:

```bash
cmake -S code/peps_cuda -B code/peps_cuda/build-a100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DPEPS_CUDA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build code/peps_cuda/build-a100 -j
ctest --test-dir code/peps_cuda/build-a100 --output-on-failure
```

Expected first failures:

- CUDA toolkit/CMake module mismatch.
- Minor `cuDoubleComplex`/atomic compile issue.
- Slurm module names differing from the template.

Fix these before touching algorithms.

## 2. Add NVTX Ranges

Add NVTX ranges around:

- projection,
- diagonal energy,
- dense Gram/apply,
- sampled-sector Gram/apply,
- later boundary absorption.

Goal: make Nsight Systems traces readable from the first real run.

## 3. Synthetic GEMM Benchmarks

Use `boundary_bucket_shapes.py` to generate shape cases, then implement a
standalone benchmark executable for:

- looped cuBLAS,
- strided batched GEMM,
- grouped GEMM,
- cuBLASLt,
- cuTENSOR.

Do this before writing custom boundary kernels.

## 4. Boundary-MPS Data Structures

Add C++/CUDA types for:

```text
BoundaryTensor {
  left_chi, right_chi, vertical_dim
  offset, strides
}

BoundaryMPS {
  row_length
  tensors
  storage
}
```

Keep compression as an interface:

```text
compress_boundary(boundary, target_chi, cutoff)
```

Initial compression can be CPU/tiny or cuSOLVER-backed; the absorption
benchmarks should not depend on final compression yet.

## 5. Single-Layer Boundary Absorption

Implement projected row absorption first:

```text
B_j[aL,aR,n] * A_j[n,e,s,w]
  -> C_j[(aL,w),(aR,e),s]
```

Acceptance:

- With no truncation and tiny sizes, matches exact CPU contraction.
- With increasing `Dc`, converges toward exact CPU contraction on `2x2`/`3x3`.

## 6. Horizontal Environments

Build row left/right environments from the single-layer boundary data.

Acceptance:

- `O` from boundary environments matches exact CPU log-gradient rows on tiny
  lattices.
- Horizontal `E` buckets reproduce exact local energies for Heisenberg/TFI.

## 7. Compact O Direct From Environments

Stop constructing dense `O` for the GPU path. Build only:

```text
Ocompact[sample][site][within_virtual_slice]
```

Then form:

```text
T = O O^dagger
theta_dot = -O^dagger x
```

using the sampled-sector Gram/apply kernels or library-backed tiled variants.

## 8. Direct Sampling

Implement Appendix-B row sampling:

- refresh double-layer boundaries,
- build conditional row density matrices,
- sample sites sequentially within a row,
- update sampled top boundary,
- return `sample`, `logpc`, and environment version.

Acceptance:

- Exact-enumeration distribution match on tiny systems.
- Stable importance weights when using stale double-layer environments.

## 9. Multi-GPU Sample Sharding

Only after single-GPU correctness:

- one rank per GPU,
- shard samples,
- local `E/O` work,
- allreduce Gram or parameter update.

Acceptance:

- 2 GPUs produce the same update as 1 GPU within FP64 tolerance on a fixed sample
  set.

## 10. Optimization Passes

Run in this order:

1. Remove host-device copies from the hot loop.
2. Remove repeated allocations.
3. Replace tiny kernel launch loops with grouped GEMM or CUDA Graphs.
4. Tune contraction layout/transposes.
5. Tune direct Gram tiling/reuse.
6. Only then write CuTe/CUTLASS kernels for stubborn fixed shapes.
