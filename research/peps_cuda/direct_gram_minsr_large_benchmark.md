# Direct-Gram minSR Larger CPU Benchmark

Date: 2026-05-15

## Setup

- Scenario: real `4x4` Heisenberg PEPS
- Bond dimension: `D=3`
- Samples per repeat: `N_s=1024`
- Repeats: `5` warmed repeats
- Warmup: same scenario with `N_s=8`
- Execution: single Julia process, `BLAS.set_num_threads(1)`, no Julia threading
- PEPS initialization: `multiply_algebraic_spectrum!(alpha=3.0)`
- Output directory:
  `research/peps_cuda/profiles/julia_cpu/direct_minsr_d3_Ns1024_R5`

The run completed in `215 s` including Julia compilation, warmup, and all five
dense/direct repeat pairs. Process peak RSS reported by `/usr/bin/time -l` was
about `1.99 GB`.

## Total Runtime

Warmed repeats:

| repeat | dense (s) | compact direct (s) | speedup |
|---:|---:|---:|---:|
| 1 | 5.7559 | 5.4733 | 1.0516x |
| 2 | 5.7988 | 5.4667 | 1.0608x |
| 3 | 5.7506 | 5.4816 | 1.0491x |
| 4 | 5.7559 | 5.4735 | 1.0516x |
| 5 | 5.7893 | 5.4665 | 1.0591x |

Aggregate:

- dense mean: `5.7701 s`, sd `0.0222 s`
- compact direct mean: `5.4723 s`, sd `0.0062 s`
- mean speedup: `1.054x`
- total wall-time reduction: `5.16%`
- max `theta_dot` relative error: `9.36e-14`
- max `T` relative error: `1.14e-15`

## Stage Breakdown

Mean over the five warmed repeats:

| stage | dense time | compact time | note |
|---|---:|---:|---|
| sampling | `2.786 s` | `2.712 s` | unchanged algorithmically |
| vertical envs | `0.856 s` | `0.895 s` | same operation, noise-level drift |
| energy | `0.925 s` | `0.899 s` | same operation |
| horizontal envs | `0.285 s` | `0.279 s` | same operation |
| `O_k` gradients | `0.430 s` | `0.276 s` | changed representation |
| solver / direct solve | `0.235 s` | `0.181 s` | dense path includes `dense_T` in solver section |
| direct `T` formation | n/a | `0.037 s` | compact sample-space Gram |

The `O_k` stage itself improved by:

- time: `1.56x` faster
- time reduction: `35.8%`
- allocations: `711 MiB -> 495 MiB`
- allocation reduction: `30.4%`

The total speedup is smaller because sampling and contractions dominate the
runtime at this size.

## Memory

Parameter counts:

- dense parameter layout: `1152`
- compact sampled-sector layout: `576`
- compact fraction: `0.5`

For `N_s=1024`:

| object | FP64 dense | FP64 compact |
|---|---:|---:|
| sampled `O` rows | `9.0 MiB` | `4.5 MiB` |
| sample-space Gram `T` | `8.0 MiB` | same |

Equivalent FP32 row storage:

| object | FP32 dense | FP32 compact |
|---|---:|---:|
| sampled `O` rows | `4.5 MiB` | `2.25 MiB` |
| sample-space Gram `T` | `4.0 MiB` | same |

Whole-run Julia allocations changed only modestly:

- dense mean allocations: `12.22 GB`
- compact direct mean allocations: `12.09 GB`
- whole-run allocation reduction: `1.04%`

This is expected: ITensor sampling, environment, and energy allocations dominate
the total allocation profile. The relevant improvement is local to sampled
Jacobian construction and memory traffic.

## Interpretation

For this larger single-core CPU run, the compact sampled-sector path gives a
stable `~5%` end-to-end speedup and a much clearer `~36%` speedup inside the
`O_k` stage. This is less dramatic than the tiny warmed `4x4 D=2` cases because
sampling and PEPS contractions dominate more strongly once `N_s` is large.

The result is still useful as a first showable optimization: it is exactly
regression-tested against the dense Julia path, reduces sampled-Jacobian storage
by `2x` for spin-1/2, and demonstrates a real end-to-end gain without changing
the physics method.
