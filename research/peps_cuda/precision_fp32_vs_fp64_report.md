# FP32 vs FP64 Fixed-Sample Precision Check

Date: 2026-05-15

## Setup

- Scenario: real `4x4` Heisenberg PEPS
- Bond dimension: `D=3`
- Samples: `N_s=1024`
- Comparison: compact sampled-sector minSR path
- Execution: single process, `BLAS.set_num_threads(1)`
- Sample set: generated once from the FP64 PEPS, then reused for FP64 and FP32
- FP32 PEPS parameters: rounded from the FP64 PEPS parameters
- Initialization: `multiply_algebraic_spectrum!(alpha=3.0)`
- Output directory:
  `research/peps_cuda/profiles/julia_cpu/precision_fp32_vs_fp64_D3_Ns1024`

This intentionally isolates arithmetic precision from Monte Carlo noise. Native
FP32 sampling may draw different samples because tiny probability changes can
cross random thresholds; that should be tested separately.

## Main Result

After warmup, on the fixed sample set:

| quantity | relative error vs FP64 | max absolute error |
|---|---:|---:|
| `logpsi` | `5.25e-8` | `9.44e-6` |
| `E_loc` | `1.26e-6` | `2.47e-4` |
| normalized weights | `2.40e-6` | `1.74e-5` |
| sample-space Gram `T` | `1.47e-6` | `3.68e-2` |
| final `theta_dot` | `4.59e-5` | `7.58e-5` |

Energy mean absolute error:

- `2.97e-7`

This is a good early sign for FP32. The final update is more sensitive than
`logpsi`, `E_loc`, and `T`, which is expected because the minSR solve can
amplify perturbations in `T` and the centered energy vector.

## Runtime And Allocations

These timings are CPU/Julia reference timings, not a GPU prediction:

| path | elapsed | allocated |
|---|---:|---:|
| FP64 compact | `3.399 s` | `7.39 GB` |
| FP32 compact | `3.168 s` | `7.16 GB` |

FP32 was about `1.07x` faster in this reference run and allocated about `3.2%`
less total memory. The small allocation reduction is because ITensor
environment and energy temporaries dominate the total allocation profile.

Stage-level examples:

| stage | FP64 | FP32 |
|---|---:|---:|
| vertical envs | `1.52 s`, `3.81 GiB` | `1.45 s`, `3.72 GiB` |
| energy | `0.923 s`, `1.57 GiB` | `0.842 s`, `1.50 GiB` |
| compact `O_k` | `0.268 s`, `495 MiB` | `0.255 s`, `481 MiB` |
| direct `T` | `38.1 ms`, `103 MiB` | `24.1 ms`, `57.7 MiB` |
| direct solve | `195 ms`, `24.6 MiB` | `182 ms`, `35.3 MiB` |

Process peak RSS for the full run, including compilation and warmup, was about
`2.11 GB`.

## Memory Model

For `N_s=1024` and compact sampled-sector rows:

| object | FP64 | FP32 |
|---|---:|---:|
| compact sampled rows | `4.5 MiB` | `2.25 MiB` |
| sample-space Gram `T` | `8.0 MiB` | `4.0 MiB` |

For the dense sampled row layout, these row numbers would be doubled again for
spin-1/2.

## Interpretation

This supports using FP32 as a serious default candidate, especially for the
memory-bound CUDA implementation. I would not yet claim that FP64 can be ignored
entirely:

- the final update error, `4.6e-5`, is much larger than the raw `logpsi` error;
- minSR conditioning can vary during optimization;
- native FP32 sampling may diverge from FP64 sampling, which this fixed-sample
  test intentionally avoids;
- CSL/complex and larger `D` cases still need the same check.

Practical next position: implement CUDA with an FP32-first path, but keep a
debug/validation FP64 path or at least FP64 accumulation/solve option for
conditioning studies and thesis-quality error plots.
