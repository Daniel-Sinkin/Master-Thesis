# Ozaki / Ozaki-II Precision Notes

Question: is Ozaki-style FP64 emulation useful for this PEPS CUDA project on
A100/H100/GH200/H200, or mainly for Blackwell?

Short answer:

- Near-term A100/H100/GH200: keep Ozaki out of the first implementation path.
  Native FP64 and FP64 tensor-core GEMM are available, and the PEPS bottleneck is
  likely contraction scheduling, irregular/batched GEMM shape efficiency,
  compression traffic, and dense-`O` memory pressure.
- H100/H200: Ozaki-II is worth a benchmark only for large, square-ish,
  compute-bound GEMM buckets where native `cublasZgemm`/`cublasDgemm` is already
  the measured bottleneck and accuracy requires FP64-like accumulation.
- Blackwell or consumer/AI GPUs with weak native FP64: Ozaki/Ozaki-II becomes
  much more interesting, but the main code should not target Blackwell.
- FP32/TF32/mixed precision is likely higher ROI for this thesis than Ozaki:
  it halves dense/compact `O` memory immediately and is easier to justify with
  energy/TDVP-error regression tests.

## What Ozaki Does

The original Ozaki scheme decomposes FP64 matrix multiplication into several
low-precision matrix multiplications plus high-precision reconstruction. The
useful property is that GEMM can run on fast tensor-core-like units while
recovering FP64-level accuracy for dense GEMM.

Ozaki-II changes the decomposition: it uses modular integer arithmetic and the
Chinese Remainder Theorem, reducing the number of GEMMs relative to the original
slice-cross-product scheme. The Ozaki-II paper reports FP64-emulation
throughput on GH200 of roughly `56.6-80.2 TFLOP/s`, compared with measured
native FP64 tensor-core DGEMM around `~60 TFLOP/s` in their setup for large
matrices.

## Why It Is Not A First-Pass PEPS Optimization

The PEPS workload is not just one large DGEMM:

- Boundary-MPS row absorption becomes many shape buckets, including small and
  medium GEMMs.
- Compression/SVD/eigendecomposition and tensor reshapes can dominate traffic.
- `E` generation has many small support-dependent contractions and diagonal
  shortcuts.
- `O`/minSR memory can dominate before arithmetic does.
- Sampling is sequential within each sample row and parallel mostly across
  samples.

Ozaki-II helps when the hot loop is dominated by large GEMMs. It does not help
vector arithmetic, indexing, kernel launch overhead, SVD workspace, or memory
capacity. The Ozaki-II paper's own breakdown says small problems suffer from
kernel-launch and conversion overhead, while large matrices amortize those
costs.

## H100/H200/GH200 Assessment

Hopper already has strong native FP64:

- H100/H200-class FP64 tensor-core peak is about `67 TFLOP/s`.
- GH200 in the Ozaki-II paper reaches at most a modest speedup over native FP64
  for large square GEMMs, not an order-of-magnitude win.
- JUPITER Booster documentation currently points to GH200/H100-class GPUs with
  96 GB HBM3, not H200. That makes memory footprint a bigger immediate limiter
  than raw FP64 GEMM throughput.

Practical thesis stance:

- Use native FP64 as the correctness baseline.
- Use complex FP32 as the first performance baseline.
- Evaluate TF32/FP32 tensor-core boundary contractions with iterative correction
  or residual checks before considering Ozaki.
- Put Ozaki-II behind a backend abstraction only if profiling shows a few large
  FP64 GEMM buckets dominate wall time.

## Where It Could Fit

Candidate backend slot:

```text
BoundaryGemmBackend:
  NativeFp64Cublas
  NativeFp32CublasLt
  Tf32CublasLt
  Cutensor
  Ozaki2Experimental
```

Eligibility checks before trying Ozaki-II:

- GEMM bucket has large enough `M,N,K` to amortize slicing/conversion.
- The same bucket repeats many times per optimization step.
- The bucket is compute-bound in Nsight Compute, not memory/launch-bound.
- The output tolerance cannot be met by FP32/TF32 plus normalization/regression
  checks.
- Extra workspace for decomposed inputs fits comfortably in HBM.

## Precision Regression Strategy

For each benchmark fixture, compare:

- energy mean and variance,
- `logpsi` relative error,
- `E_loc` relative error,
- sampled-sector Gram spectrum,
- minSR direction cosine similarity and relative norm error,
- one optimizer step's loss/energy change.

Decision thresholds should be physics-facing. If FP32 changes low-level tensor
entries but preserves energy and TDVP direction within Monte Carlo noise, FP32 is
probably acceptable. If FP32 destabilizes late-time or high-`D` contractions, try
mixed precision or selective FP64 before Ozaki.

## Current Recommendation

Do not spend first implementation time on Ozaki kernels. Preserve a GEMM backend
interface so Ozaki-II/cuBLAS FP64 emulation can be plugged in later, and mention
it in the thesis as a plausible future backend for Blackwell or large
compute-bound H100/GH200 GEMM buckets.
