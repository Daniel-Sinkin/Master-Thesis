# cuBLAS/cuTENSOR Benchmark Plan For Boundary Buckets

This note turns the boundary-MPS shape buckets into concrete library baselines.
The first goal is not to win; it is to avoid writing custom kernels before the
library baselines have been measured on A100 and GH200/H100.

## Candidate Backends

### Looped cuBLAS

Use plain `cublasZgemm`/`cublasDgemm` calls in a loop over sites/shapes.

Why:

- Simplest correctness baseline.
- Exposes how bad launch/API overhead is.

Expected:

- Often poor for many small/medium contractions.
- Useful as a floor.

### Strided Batched GEMM

Use `cublasZgemmStridedBatched` or real equivalents when a shape bucket has
identical `M,N,K,lda,ldb,ldc`.

Why:

- Very natural for interior sites where true dimensions match.
- Avoids pointer-array setup.

Expected:

- Good for large identical buckets.
- Not enough for edge/interior mixtures or compression-varying `chi`.

### Grouped GEMM

Use cuBLAS 12.5+ grouped GEMM:

```text
cublas<t>gemmGroupedBatched
cublasGemmGroupedBatchedEx
```

Why:

- Supports variable matrix sizes/transposes/scales in one grouped launch.
- PEPS boundary absorption has exactly this pattern: many related contractions
  with edge/interior shape variation.

Watch-outs:

- Pointer arrays and group metadata can introduce host/device setup overhead.
- Use Nsight Systems to check for hidden `cudaMemcpyAsync` or setup costs.
- Grouped GEMM may not always beat strided batched GEMM for uniform interior
  buckets.

### cuBLASLt

Use cuBLASLt for dense Gram, larger contraction buckets, and autotuning.

Why:

- Better heuristic/tuning surface than classic cuBLAS.
- Useful for testing tensor-core eligible modes later.

Expected:

- Strong baseline for dense/sampled-sector Gram once data is laid out as GEMM.
- More setup complexity; cache descriptors/plans.

### cuTENSOR

Use cuTENSOR for contractions where avoiding explicit transposes matters more
than reducing everything to GEMM.

Why:

- Direct tensor descriptors and strides.
- JIT/plan cache can amortize repeated contraction descriptors.

Expected:

- Potentially strong for awkward site-gradient or row-absorption variants.
- Needs careful plan reuse; first-call JIT/planning time should be excluded from
  steady-state timings.

## First Shape Manifest

Generate:

```bash
python3 code/peps_cuda/tools/boundary_bucket_shapes.py \
  --lx 16 --ly 16 --d 8 --dc 64 --dc-double 64
```

Typical dominant rows under the simple compressed-chi model:

```text
single_absorb: M=4096, N=512,    K=8,  count=196
double_absorb: M=4096, N=262144, K=64, count=196
```

These are approximate. Real compression changes `chi`, but these shapes are good
enough to start the benchmark matrix.

## Data Layout For GEMM Baselines

Single-layer absorption:

```text
B_mat[(aL,aR), n]        M x K
A_mat[n, (w,e,s)]        K x N
C_mat[(aL,aR), (w,e,s)]  M x N
```

Double-layer absorption:

```text
B2_mat[(aL,aR), (n,n')]              M x K
M_mat[(n,n'), ((w,w'),(e,e'),(s,s'))] K x N
C2_mat[(aL,aR), ...]                 M x N
```

Store the contracted leg `K` contiguous where possible. If not, test whether
cuTENSOR beats explicit transpose + GEMM.

## Measurement Matrix

For each shape bucket:

```text
backend:
  cublas_loop
  cublas_strided_batched
  cublas_grouped
  cublaslt
  cutensor

precision:
  complex_fp64
  complex_fp32
  fp64_real_synthetic
  fp32_real_synthetic
```

Record:

- `M,N,K,count`
- elapsed time after warmup
- achieved TFLOP/s
- achieved GB/s
- CUDA API time from Nsight Systems
- workspace bytes
- whether descriptors/plans were reused

## Decision Rules

- If looped cuBLAS is slow due to launch overhead but each GEMM is efficient,
  grouped/strided batching is the next step.
- If grouped GEMM has significant setup copies, keep metadata resident and test
  CUDA Graph capture.
- If transpose/copy time dominates, try cuTENSOR with native strides.
- If all library baselines underutilize SMs on a repeated fixed shape, then
  write a CuTe/CUTLASS kernel for that shape.
- If compression/SVD dominates the timeline, stop tuning absorption and focus on
  the compression algorithm.

## Relation To Current Code

Current CUDA kernels cover:

- projection,
- diagonal energy,
- dense minSR smoke Gram/apply,
- sampled-sector minSR smoke Gram/apply.

They do not yet perform boundary-MPS absorption. The scripts and notes here are
the handoff: generate shape buckets, benchmark libraries on those buckets, then
implement the boundary layer using the winning backend per shape family.
