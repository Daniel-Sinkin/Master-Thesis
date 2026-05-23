# Direct-Gram minSR CPU Baseline

Date: 2026-05-15

## What This Changes

The paper already introduces minSR for finite PEPS: instead of solving with the
large parameter-space matrix `G = O^dagger O`, it solves in sample space with
`T = O O^dagger`, whose size is `N_s x N_s`.

This experiment is narrower. The Julia reference already uses the sample-space
minSR route, but it still constructs `O` in the full dense PEPS parameter layout.
For each sample and each site, only the physical sector selected by the sampled
configuration is nonzero. For spin-1/2, that means half of the dense row is
structural zeros.

The added path stores only the active physical sector per site and forms the
same centered/weighted `T` directly:

```text
T[s,t] = <O_s - mean(O), O_t - mean(O)>
```

with the same importance-weight scaling as `QuantumNaturalGradient.Jacobian`.
It then computes the same minSR update:

```text
raw = -solve(T, centered(E))
theta_dot = centered(O)^dagger raw
```

without materializing the dense sampled Jacobian.

## Files

- `research/peps_cuda/sources/QuantumNaturalfPEPS.jl-main/src/DirectMinsr.jl`
  implements compact sampled-sector gradients and direct sample-space Gram
  accumulation.
- `research/peps_cuda/sources/QuantumNaturalfPEPS.jl-main/src/QuantumNaturalfPEPS.jl`
  includes and exports the new reference helpers.
- `code/peps_cuda/julia_reference/test_direct_minsr.jl`
  regression-tests compact scatter and end-to-end dense-vs-direct minSR.
- `code/peps_cuda/julia_reference/benchmark_direct_minsr_cpu.jl`
  benchmarks dense and direct paths with matched RNG seeds.

## Correctness

Command:

```bash
julia --project=code/peps_cuda/julia_reference --compiled-modules=no \
  code/peps_cuda/julia_reference/test_direct_minsr.jl
```

Results:

- compact `O_k` scatter reproduces dense `O_k` for real and complex `3x2`, `D=2`.
- end-to-end direct minSR reproduces dense Julia minSR for real and complex
  `3x2`, `D=2`.
- real case: `theta_rel_err = 1.07e-15`, `T_rel_err = 3.87e-16`.
- complex case: `theta_rel_err = 1.68e-15`, `T_rel_err = 4.31e-16`.

## CPU Benchmark Notes

All benchmark PEPS are regularized with
`multiply_algebraic_spectrum!(alpha=3.0)`, in line with the paper's warning
that naive random PEPS are a poor starting point for these contractions.

The first tiny warmup pass is dominated by Julia compilation and should not be
used as a performance claim. The warmed pass is the useful number.

### Real `4x4` Heisenberg, `D=2`, `N_s=128`

Output directory:
`research/peps_cuda/profiles/julia_cpu/direct_minsr_real4x4_Ns128_warm_same`

- dense elapsed: `0.660322 s`
- direct elapsed: `0.579857 s`
- speedup: `1.14x`
- `theta_rel_err = 2.53e-13`
- `T_rel_err = 1.14e-15`
- samples, weights, and energies match exactly within recorded precision.
- dense allocations: `1.45 GB`
- direct allocations: `1.42 GB`
- process peak RSS for whole benchmark run: about `1.98 GB`

Timer detail:

- dense `log_gradients`: `49.5 ms`, `84.8 MiB`
- compact `compact_log_gradients`: `33.6 ms`, `58.0 MiB`

This is the direct structural win: less zero-sector writing and less per-sample
gradient storage. The whole iteration only improves modestly because sampling,
vertical environments, and local energy dominate at this size.

### Complex `4x4` CSL, `D=2`, `N_s=64`

Output directory:
`research/peps_cuda/profiles/julia_cpu/direct_minsr_complex4x4_Ns64_warm_same`

- dense elapsed: `0.539828 s`
- direct elapsed: `0.394966 s`
- speedup: `1.37x`
- `theta_rel_err = 3.17e-13`
- `T_rel_err = 1.14e-15`
- samples, weights, and energies match exactly within recorded precision.
- dense allocations: `887 MB`
- direct allocations: `873 MB`
- process peak RSS for whole benchmark run: about `1.85 GB`

Timer detail:

- dense `log_gradients`: `30.3 ms`, `43.0 MiB`
- compact `compact_log_gradients`: `15.7 ms`, `29.6 MiB`

## Memory Model

For spin-1/2, compact sampled-sector rows are exactly half the dense sampled
Jacobian rows. For local dimension `d`, this generalizes to roughly `1/d` of
the dense row storage.

Real `4x4`, `D=2`, `N_s=128`:

- dense FP64 rows: `294,912 B`
- compact FP64 rows: `147,456 B`
- sample-space Gram FP64: `131,072 B`
- dense FP32 rows: `147,456 B`
- compact FP32 rows: `73,728 B`
- sample-space Gram FP32: `65,536 B`

Complex `4x4`, `D=2`, `N_s=64`:

- dense ComplexF64 rows: `294,912 B`
- compact ComplexF64 rows: `147,456 B`
- sample-space Gram ComplexF64: `65,536 B`
- dense ComplexF32 rows: `147,456 B`
- compact ComplexF32 rows: `73,728 B`
- sample-space Gram ComplexF32: `32,768 B`

At these toy sizes, the Jacobian itself is not the main RSS driver. The value of
this representation should become more visible as `L`, `D`, `d`, and `N_s`
increase, and it is especially relevant for a CUDA implementation because it
removes pointless memory traffic.

## Interpretation

This is not a new replacement for minSR. It is a more compact implementation of
the same minSR algebra used by the paper and the Julia reference.

On CPU and small examples, this is already measurable but not dramatic after
proper warmup: about `1.1x` to `1.4x` total speedup in the two 4x4 tests. The
gradient substage itself improves more clearly, around `1.5x` to `1.9x`.

For CUDA, this is still a useful baseline because it clarifies the invariant:
the GPU implementation can stream or store only active physical sectors, form
the same `T`, and scatter the final update back into the full PEPS parameter
layout only once.
