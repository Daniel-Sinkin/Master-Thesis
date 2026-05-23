# Reference Alignment Plan

The C++/CUDA baseline should first match the Julia behavior before optimizing
away the expensive abstractions. The reference package is useful, but it is not
currently reproducible from `Project.toml` alone.

## Reference Setup Findings

- `QuantumNaturalfPEPS.jl` has no `Manifest.toml`.
- `QuantumNaturalGradient.jl` is unregistered; the paper's Zenodo metadata
  points to `https://github.com/danielalcalde/QuantumNaturalGradient.jl`.
- `ParallelGradient` is unregistered. In the current `QuantumNaturalGradient.jl`
  snapshot it only provides `@addprocs_and_everywhere` and `@everywhere_async`;
  the C++ scaffold includes a small local compatibility stub for harness runs.
- Current `ITensors` releases break the PEPS code at load time because
  `ITensors.AbstractMPS` and related helper symbols moved. Pinning
  `ITensors = 0.6.23` and `ITensorMPS = 0.2.2` lets the package load with
  `--compiled-modules=no`.
- With those pins the package loads, but boundary-logpsi calls can still fail
  because `deprecate_make_inds_match!` and `check_hascommoninds` are not
  available from that ITensors version. The fixture exporter records
  `logpsi_exact`; for controlled fixtures it also installs compatibility shims
  for those helpers and the old `ITensors._log_or_not_dot` symbol so boundary
  `logpsi` can be exported when indices already match.
- `QuantumNaturalGradient.__init__()` has a load/precompile bug:
  `occursin(".julia/dev/", pathof(QuantumNaturalGradient))` can receive
  `nothing`. This is an upstream reproducibility bug, not a physics issue.
- `get_logψ_and_envs` has a default-position bug for two-row systems:
  `env_top` has length one, so the default `pos=length(env_top)÷2` is zero and
  indexing fails. Fixtures use at least three rows or pass `pos=1`.
- Additional two-row fixtures pass `pos=1` and validate `logpsi`, but their
  `E/O` export records a separate `BoundsError` on `env_top[0]`. Treat those as
  reference bug records until the Julia path is patched.
- Fixture rows now include native ITensor axis labels plus explicit
  `theta_site_dims`/`theta_axis_labels` in the flattened `vec(peps)` order. This
  is needed for reconstructing `D>1` open-boundary tensors in C++ without
  guessing which virtual axis is north/east/south/west.
- Fixture rows also include `sample_row_major` because Julia's `vec(sample)` is
  column-major while the C++ scaffold stores sample spins in site-major
  `(x,y)` order.

## Fixture Classes

Use both upstream examples and generated cases:

- Product/random PEPS, `3x2`, `D=1`, Heisenberg. This catches conventions and exact
  amplitudes.
- Random real PEPS, `3x2`, `D=2`, Heisenberg. This catches ragged open-boundary
  tensor dimensions, nonzero checker samples, and horizontal/vertical support
  handling.
- Random complex PEPS, `3x2`, `D=2`, Heisenberg. This catches phase conventions
  and holomorphic-gradient paths.
- Extra generated rows: `2x3,D=2` real/complex striped samples and
  `2x2,D=3` real checker sample. These currently validate `logpsi`/boundary
  conventions and preserve the Julia two-row `E/O` bug as data.
- Rydberg-style diagonal long-range Hamiltonian. This catches the cheap diagonal
  path separately from flip contractions.
- Stale double-layer sampling fixture with stored sampling probabilities. This
  catches importance-weight normalization.

## Comparison Targets

For each fixture, save JSON-lines records for:

- PEPS tensor dimensions and flattened tensor data.
- Native and theta-order ITensor axis labels.
- Samples in both Julia `vec` order and C++ row-major order.
- Sample bitstrings and Julia sampling log-probabilities.
- `logpsi` from boundary environments and `logpsi_exact`.
- `E_loc` split by Hamiltonian bucket.
- Dense `O` row and compact sampled-sector row.
- minSR Gram, minSR direction, and weighted minSR direction.
- Julia `Base.gc_live_bytes()` plus process peak RSS from `run_with_memory.sh`.

The C++ side should compare absolute and relative error per field. Until the
reference tensors are imported exactly, only algebraic invariants are strict.
The current strict bridge imports the Julia `real_3x2_D1_zero_sample`,
`real_3x2_D2_zero_sample`, and `complex_3x2_D2_zero_sample` fixtures directly
into the C++ unit tests. The `D=2` cases explicitly transpose from Julia
column-major theta order into C++ tensor storage and back into Julia `O_k`
order, then check `logpsi`, `O_k`, `||O_k||^2`, phase conventions, and the
Pauli-normalized Heisenberg energy:

- sampled-sector Gram must match dense sparse-row Gram,
- parameter-space SR must match dual minSR on tiny systems,
- importance weights must have mean one,
- exact `logpsi` and boundary `logpsi` must agree within the contraction cutoff.
