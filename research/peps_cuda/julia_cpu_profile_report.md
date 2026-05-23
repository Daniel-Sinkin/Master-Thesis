# Julia CPU Profiling Report

Date: 2026-05-15

This profiles the Julia reference implementation on the local MacBook CPU. No
CUDA/GPU path was used. The goal is to characterize the reference code path, not
the first C++ draft.

## Harness

Script:

```text
code/peps_cuda/julia_reference/profile_reference_cpu.jl
```

Profile directories:

```text
research/peps_cuda/profiles/julia_cpu/latest_singlethread_s8
research/peps_cuda/profiles/julia_cpu/latest_threaded_s8
research/peps_cuda/profiles/julia_cpu/latest_threaded_smoke
```

Commands used:

```bash
code/peps_cuda/julia_reference/run_with_memory.sh \
  julia --threads=1 --project=code/peps_cuda/julia_reference \
  --compiled-modules=no \
  code/peps_cuda/julia_reference/profile_reference_cpu.jl \
  --samples=8 --maxiter=1 \
  --out=research/peps_cuda/profiles/julia_cpu/latest_singlethread_s8

code/peps_cuda/julia_reference/run_with_memory.sh \
  julia --threads=4 --project=code/peps_cuda/julia_reference \
  --compiled-modules=no \
  code/peps_cuda/julia_reference/profile_reference_cpu.jl \
  --samples=8 --maxiter=1 --threaded \
  --out=research/peps_cuda/profiles/julia_cpu/latest_threaded_s8
```

The upstream examples are too large to run literally as a local profiling pass:

- `examples/heisenberg_multithreaded.jl`: `4x4,D=2,Ns=1000,maxiter=10`.
- `examples/CSL.jl`: `4x4,D=2,Ns=1000,maxiter=4000`.

So the harness profiles the same example workloads with `Ns=8,maxiter=1` and
keeps the parameters configurable. That is enough to identify stage-level CPU
hotspots without spending hours on a MacBook.

## Scenarios

- `example_heisenberg_4x4_d2`: reduced version of
  `examples/heisenberg_multithreaded.jl`.
- `example_csl_4x4_d2`: reduced version of `examples/CSL.jl`.
- `synthetic_real_3x2_d2_heisenberg`.
- `synthetic_complex_3x2_d2_heisenberg`.
- `synthetic_real_3x3_d2_tfi`.

## Single-Thread Results

All runs use `Ns=8,maxiter=1`.

| Scenario | Elapsed | Integrator Timer | Alloc In Integrator | Main Bottleneck |
| --- | ---: | ---: | ---: | --- |
| Heisenberg `4x4,D=2` | `71.9 ms` | `69.0 ms` | `92.4 MiB` | sampling/envs |
| CSL `4x4,D=2` | `91.7 ms` | `91.1 ms` | `112 MiB` | sampling + four-body energy |
| Real `3x2,D=2` Heisenberg | `84.3 ms` | `76.2 ms` | `19.3 MiB` | sampling + double layer |
| Complex `3x2,D=2` Heisenberg | `25.5 ms` | `24.6 ms` | `19.6 MiB` | sampling + vertical envs |
| Real `3x3,D=2` TFI | `33.9 ms` | `33.2 ms` | `36.3 MiB` | sampling + gradients |

The `4x4` examples are the useful anchors:

### Heisenberg `4x4,D=2`

Timer path:

```text
integrator:        69.0 ms, 92.4 MiB
NaturalGradient:   68.4 ms, 91.3 MiB
Oks_and_Eks:       68.1 ms, 91.3 MiB
inner Oks/Eks:     63.1 ms, 86.4 MiB
sampling:          29.9 ms, 46.2 MiB
energy:            13.3 ms, 11.6 MiB
vertical_envs:     11.4 ms, 16.3 MiB
log_gradients:      4.0 ms,  5.3 MiB
double_layer_envs:  5.0 ms,  4.9 MiB
```

`Oks_and_Eks` is effectively the whole iteration. Sampling alone is about
`43%` of the measured integrator time and `50%` of allocations.

### CSL `4x4,D=2`

Timer path:

```text
integrator:        91.1 ms, 112 MiB
NaturalGradient:   90.3 ms, 111 MiB
Oks_and_Eks:       89.0 ms, 111 MiB
inner Oks/Eks:     83.7 ms, 105 MiB
sampling:          34.8 ms, 47.4 MiB
energy:            22.4 ms, 20.6 MiB
  fourbody:        17.1 ms, 11.4 MiB
vertical_envs:      8.8 ms, 16.6 MiB
precomp_sHpsi:      8.7 ms, 11.1 MiB
horizontal_envs:    5.5 ms,  4.4 MiB
log_gradients:      3.1 ms,  5.4 MiB
double_layer_envs:  5.3 ms,  5.1 MiB
```

The CSL example makes the Hamiltonian path much more visible. Four-body energy
contractions dominate the energy part, and precomputing flipped elements is
nontrivial.

## Threaded Results

All runs use `Ns=8,maxiter=1` and `julia --threads=4`.

| Scenario | Single Thread | Threaded | Rough Speedup |
| --- | ---: | ---: | ---: |
| Heisenberg `4x4,D=2` | `71.9 ms` | `28.8 ms` | `2.5x` |
| CSL `4x4,D=2` | `91.7 ms` | `56.1 ms` | `1.6x` |
| Real `3x2,D=2` Heisenberg | `84.3 ms` | `9.75 ms` | noisy/small case |
| Complex `3x2,D=2` Heisenberg | `25.5 ms` | `11.8 ms` | `2.2x` |
| Real `3x3,D=2` TFI | `33.9 ms` | `17.1 ms` | `2.0x` |

Important limitation: `Oks_and_Eks_threaded` does not pass the `timer` keyword
down into `Ok_and_Ek`, so the detailed substage timing is lost in threaded mode.
The threaded TimerOutput mostly shows:

```text
integrator
NaturalGradient
Oks_and_Eks
double_layer_envs
solver
```

That is a reference-code instrumentation bug/limitation.

## Memory Notes

The process peak from `/usr/bin/time -l` is dominated by Julia/ITensors package
load and JIT:

- single-thread profiling process peak memory footprint: about `1.97 GB`.
- threaded profiling process peak memory footprint: about `2.07 GB`.

Per-scenario live bytes after a profiled run were much smaller but still
allocation-heavy:

- Heisenberg `4x4,D=2`: `gc_live_after` about `364 MB`.
- CSL `4x4,D=2`: `gc_live_after` about `412-413 MB`.

For these tiny `Ns=8` runs, allocation pressure is already tens to hundreds of
MiB per iteration. At real `Ns=1000`, dense `Oks` dominates even more strongly.

## Multiprocess Path

I added a `--multiproc` mode to the harness, but local multiprocess profiling is
not yet a clean data source:

- Julia's standard `Profile` on the main process does not capture worker stacks
  as a useful unified profile.
- The first worker attempt failed because workers did not activate the local
  harness project and could not find the unregistered `QuantumNaturalGradient`.
- The second attempt exposed Julia world-age friction around activating `Pkg` on
  workers. The script now uses `Base.invokelatest(Pkg.activate, project)`, but I
  did not treat multiprocess timing as a validated result in this report.

For thesis profiling, prefer single-process/threaded TimerOutput locally and
use OS-level sampling or per-worker profile files if the distributed Julia path
must be characterized further.

## Interpretation

The CPU profile supports the CUDA plan:

1. `Oks_and_Eks` is the optimization target, not the solver.
2. Sampling and environment construction are major CPU hotspots.
3. CSL/four-body terms make the energy path much more important than plain
   Heisenberg.
4. Dense `Oks` allocation is already heavy at toy sample counts.
5. Threading helps, but the current threaded implementation hides the substage
   timing and still materializes dense `Oks`.

For C++/CUDA, the biggest expected wins are resident boundary/sample buffers,
bucketed contraction libraries/kernels, and direct sampled-sector Gram
accumulation instead of dense `Oks`.
