# Julia Reference Code Review

Scope: `QuantumNaturalfPEPS.jl` plus the companion
`QuantumNaturalGradient.jl` snapshot. This is a semantic reference, not an
optimized implementation.

## What It Does Well

- The algorithm stages are cleanly separated:
  `get_sample -> get_logψ_and_envs -> get_all_horizontal_envs -> get_Ek/get_Ok`
  in `src/Ok_and_Ek.jl:7-22`.
- The code uses boundary environments rather than fully exact contraction for
  production paths. That is the right algorithmic shape for finite PEPS.
- `Ek.jl` already buckets Hamiltonian terms by geometry:
  horizontal, four-body/vertical, longer-horizontal, and fallback exact
  contraction. This is exactly what CUDA needs as separate contraction buckets.
- `Ok.jl` writes zeros for non-sampled physical sectors, which exposes the
  sampled-physical-sector memory reduction used by Wu/Nys.
- `Distributed/Oks_and_Eks.jl:2-5` has the importance-weight formula in a stable
  log-sum-exp style; this is already mirrored in C++.
- The solver in `QuantumNaturalGradient/src/solver/solver.jl:20-24` chooses the
  sample-space `T = O O^dagger` solve when `Ns < Np`, i.e. minSR.
- The examples correctly avoid nested BLAS oversubscription by setting worker
  BLAS threads to one and reserving main-process BLAS threads for double-layer
  construction.

## Highest-ROI Improvements

1. Replace dense `Oks = Matrix(..., length(peps), sample_nr)` in
   `Distributed/Oks_and_Eks.jl:83` with compact sampled-sector rows or direct
   Gram accumulation. Dense `O` is the memory wall.
2. Lower boundary row absorption to explicit shape buckets and call
   cuBLASLt/grouped GEMM/cuTENSOR. ITensor abstraction is excellent for
   research, but hides the layout and batching decisions needed for GPU
   performance.
3. Keep PEPS tensors, projected rows, environments, samples, `E`, and compact
   `O` resident on GPU. The Julia reference has no CUDA dependency and does not
   use `CuArray`; any GPU use through ITensor would be indirect and currently is
   not set up in this repo.
4. Precompute Hamiltonian flip buckets once per Hamiltonian/sample-shape class
   instead of reconstructing dynamic dictionaries and tuple keys per sample.
5. Add a real reference fixture/Manifest. The repo currently needs manual pins
   and a local compatibility stub to load.
6. Separate "physics defaults" from "benchmark defaults". The examples are good
   scientific demos but too heavy as regression tests.

## Reproducibility And Correctness Issues Found

- `QuantumNaturalfPEPS.jl` has no `Manifest.toml`, and two dependencies are
  unregistered. This makes exact reproduction fragile.
- `QuantumNaturalGradient.__init__()` calls
  `occursin(".julia/dev/", pathof(QuantumNaturalGradient))`; during dependency
  precompile/load, `pathof` can be `nothing`, which crashes precompilation.
- Current `ITensors` breaks the package; the reference harness pins
  `ITensors=0.6.23` and `ITensorMPS=0.2.2`.
- `get_logψ_and_envs` has a two-row default bug:
  `pos=length(env_top)÷2` gives zero when `length(env_top)==1`, causing
  `env_top[0]`.
- Even when `get_logψ_and_envs(...; pos=1)` is forced, the two-row fixture rows
  now show an `E/O` path indexing bug:
  `BoundsError: attempt to access 1-element Vector{QuantumNaturalfPEPS.Environment} at index [0]`.
  The fixture harness preserves these rows for `logψ`/boundary validation and
  as a reference bug record.
- `Ek.jl:27` has `insert(vetr, flip_term)`, a typo that would break the
  `vertical=true` path. The production path calls `sort_dict(..., vertical=false)`
  in `Ek.jl:264`, apparently avoiding it by routing vertical terms into the
  four-body bucket.
- `NaturalGradient.jl:148-149` computes `Eks_eff = -(centered(J) * θdot)` and
  immediately overwrites it with `centered(Es)`, so `tdvp_error` is suspect.
  `tdvp_relative_error` looks like the more meaningful diagnostic.
- Several functions allocate dictionaries/vectors with `Any` keys and dynamic
  tuple structures inside sample loops. That is fine for clarity but poor for
  GPU and difficult for type inference.

## GPU/Resource Critique

- There is no explicit GPU path in the project dependencies. `ITensors` and
  `NDTensors` can have GPU-related ecosystem support in other setups, but this
  repo does not configure CUDA, cuTENSOR, or GPU arrays.
- The sample loop parallelizes over CPU threads/processes. That exposes
  embarrassingly parallel samples, but each sample still performs ITensor-level
  contractions sequentially.
- Dense `O` storage scales as `Ns * Lx * Ly * d * D^4` complex numbers. At
  `32x32, D=8, d=2, Ns=5000`, dense complex FP64 `O` alone is about `625 GiB`;
  compact sampled-sector FP64 is still about `312.5 GiB`.
- The current dictionary-based Hamiltonian term expansion is not GPU-friendly.
  A CUDA baseline should convert it into flat records sorted by support kind.

## CPU Profiling Findings

See `julia_cpu_profile_report.md` for the detailed run logs. In the local CPU
profiles with `Ns=8,maxiter=1`:

- `Oks_and_Eks` is effectively the whole measured iteration: about `98%` of the
  Heisenberg `4x4,D=2` integrator time and about `98%` of the CSL
  `4x4,D=2` integrator time in the single-thread run.
- Heisenberg `4x4,D=2` single-thread split:
  sampling `29.9 ms`, energy `13.3 ms`, vertical environments `11.4 ms`,
  log-gradients `4.0 ms`, double-layer refresh `5.0 ms`.
- CSL `4x4,D=2` single-thread split:
  sampling `34.8 ms`, energy `22.4 ms`, vertical environments `8.8 ms`,
  precomputed flipped elements `8.7 ms`, log-gradients `3.1 ms`.
- Threading gives useful but limited speedup on the reduced examples:
  about `2.5x` for Heisenberg and `1.6x` for CSL on `julia --threads=4`.
- The threaded implementation does not pass `timer` down to `Ok_and_Ek`, so it
  loses the detailed `sampling/energy/log_gradients` substage timing.

## What To Preserve

- The exact stage ordering and returned fields: `Oks`, `Eks`, `logψs`,
  `samples`, `weights`, `contract_dims`.
- The importance-weight convention and normalization.
- The stale/asynchronous double-layer idea, but implement it with explicit
  environment versioning and profiler ranges.
- The geometry buckets. Even if we rewrite the kernels, this is the right
  decomposition.
- The minSR automatic switch. For real-time runs with `Ns ~ Np`, keep a
  parameter-space or iterative fallback as in the reference.
