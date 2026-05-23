# Precision Decision Matrix

The first production code should make precision a controlled experiment, not a
compile-time accident. The main question is where FP64 is scientifically needed
and where FP32/TF32/lower storage is justifiable.

## Baseline Roles

| Precision | Role | First Use |
| --- | --- | --- |
| complex FP64 | correctness oracle and thesis reference | Julia fixture parity, small CPU/GPU tests, final accuracy comparisons |
| complex FP32 | primary performance candidate | boundary contractions, sampled-sector `O`, Gram/apply paths |
| TF32 | optional GEMM backend | boundary-MPS absorption buckets after FP32 fixtures pass |
| BF16/FP16 storage | later memory experiment | storing sampled-sector rows or stale environments, not first physics result |
| FP8/Ozaki-II | future backend | only for large compute-bound GEMM buckets or weak-FP64 hardware |

## What To Measure

For each precision variant, record:

- `logpsi` absolute and relative error on fixed samples,
- local energy mean, variance, and max sample error,
- sampled-sector Gram Frobenius error and eigenvalue drift,
- minSR direction cosine similarity to FP64,
- minSR direction relative norm error,
- one-step energy/loss change,
- sampler log-probability error and importance-weight variance,
- contraction truncation error or discarded weight,
- memory footprint by stage and peak HBM/RSS.

## Suggested Acceptance Bands

These are starting bands, not final physics criteria:

| Quantity | Barely Acceptable | Good |
| --- | ---: | ---: |
| `logpsi` relative error | `1e-4` | `1e-6` |
| `E_loc` relative error | `1e-3` | `1e-5` |
| Gram relative Frobenius error | `1e-3` | `1e-5` |
| minSR direction cosine vs FP64 | `>0.999` | `>0.99999` |
| one-step energy drift beyond MC noise | not systematic | indistinguishable |

If Monte Carlo noise is larger than deterministic precision error, FP32 is
probably defensible for that regime. If the deterministic precision error
dominates or destabilizes minSR, try selective FP64 before exotic emulation.

## Likely Policy

1. Keep PEPS parameters and boundary environments FP64 for reference runs.
2. Test complex FP32 for projected tensors, sampled-sector `O`, Gram/apply, and
   maybe boundary environments.
3. Keep reductions/normalization scalars in FP64 when cheap: log-sum-exp for
   importance weights, Gram diagonal shifts, and residual/error summaries.
4. Use cuBLASLt TF32 only for boundary GEMM buckets with explicit residual
   checks. Do not use `--use_fast_math` in the correctness baseline.
5. Defer BF16/FP16/FP8 storage until direct Gram and boundary reuse work; lower
   storage precision is not useful if the algorithm is still materializing the
   wrong object.

## Thesis Argument

The memory argument is immediate:

- `16x16,D=8,Ns=2000`: compact sampled-sector `O` is about `31.37 GiB` in
  complex FP64 and `15.68 GiB` in complex FP32.
- `32x32,D=8,Ns=5000`: compact sampled-sector `O` is about `313.06 GiB` in
  complex FP64 and `156.54 GiB` in complex FP32, still too large for one
  documented JUPITER GH200 GPU.
- On an H200 SXM-style 141 GB HBM target, the same `32x32,D=8,Ns=5000`
  compact sampled-sector `O` is still too large in FP32 (`156.54 GiB`) before
  workspaces and buffers. H200 helps, but does not remove the need for direct
  Gram/batching.

Therefore FP32 is useful but not sufficient by itself. The real thesis
performance story must combine precision reduction with direct/streamed Gram
accumulation, boundary reuse, and sample sharding.
