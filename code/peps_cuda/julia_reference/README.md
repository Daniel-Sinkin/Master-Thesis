# Julia Reference Alignment Harness

This directory is for checking the standalone C++/CUDA scaffold against the
Julia reference code from `QuantumNaturalfPEPS.jl` and
`QuantumNaturalGradient.jl`.

The Julia packages are not registered reproducibly as shipped by the PEPS repo:

- `QuantumNaturalGradient.jl` is an unregistered dependency; the Zenodo record
  for the paper points to `https://github.com/danielalcalde/QuantumNaturalGradient.jl`.
- `ParallelGradient` is also unregistered. In the downloaded
  `QuantumNaturalGradient.jl` snapshot it is only used for two distributed
  helper macros, so this harness provides a small local compatibility stub.
- Current `ITensors` releases moved MPS/MPO APIs. The PEPS package loads against
  the older `ITensors` `0.6.x` API family, so this harness pins `ITensors` to
  `0.6.23` and `ITensorMPS` to `0.2.2`.
- `QuantumNaturalGradient.__init__()` calls `occursin(..., pathof(...))`, which
  can fail during dependency precompilation when `pathof` is `nothing`.
  Run reference scripts with `--compiled-modules=no` unless that upstream bug is
  patched.
- `export_reference_fixtures.jl` adds harness-only shims for
  `deprecate_make_inds_match!`, `check_hascommoninds`, and the old internal
  `ITensors._log_or_not_dot` symbol when the pinned ITensors version does not
  provide them. This is only for controlled fixtures whose MPS indices already
  match.

Setup:

```bash
julia --project=code/peps_cuda/julia_reference -e 'using Pkg; Pkg.develop(path="code/peps_cuda/julia_reference/ParallelGradient"); Pkg.develop(path="research/peps_cuda/sources/QuantumNaturalGradient.jl-main"); Pkg.develop(path="research/peps_cuda/sources/QuantumNaturalfPEPS.jl-main"); Pkg.add(Pkg.PackageSpec(name="ITensors", version="0.6.23")); Pkg.add(Pkg.PackageSpec(name="ITensorMPS", version="0.2.2")); Pkg.resolve()'
```

Load check:

```bash
julia --project=code/peps_cuda/julia_reference --compiled-modules=no -e 'using QuantumNaturalfPEPS; println("loaded")'
```

Export fixtures:

```bash
code/peps_cuda/julia_reference/run_with_memory.sh \
  julia --project=code/peps_cuda/julia_reference --compiled-modules=no \
  code/peps_cuda/julia_reference/export_reference_fixtures.jl \
  code/peps_cuda/julia_reference/fixtures/reference_fixtures.jsonl
```

Validate the JSON-lines artifact:

```bash
python3 code/peps_cuda/julia_reference/validate_reference_fixtures.py \
  code/peps_cuda/julia_reference/fixtures/reference_fixtures.jsonl
```

Summarize the fixture rows:

```bash
python3 code/peps_cuda/julia_reference/summarize_reference_fixtures.py \
  code/peps_cuda/julia_reference/fixtures/reference_fixtures.jsonl
```

The fixture export is intentionally JSON-lines. It is easy to append new cases,
diff regression output, and consume from C++/Python without adding a dependency.
Rows include both native ITensor axis labels and theta-order axis labels, plus a
`sample_row_major` field for the C++ site-major convention. The validator uses
those labels to reconstruct small fixtures by explicit link enumeration,
including `D=1`, `D=2`, and the tiny `D=3` checker row. It rechecks `logpsi`
for every reconstructed row and, when Julia successfully exports `O_k`,
rechecks the `O_k` prefix and `||O_k||^2`.

Current fixture count:

- 1 metadata row.
- 8 log-amplitude rows.
- 5 rows with Julia `E/O` output.
- 3 extra rows that intentionally preserve a two-row Julia `E/O` indexing
  failure while still validating `logpsi` and boundary environments.

Full examples:

The reference repo has two examples:

- `examples/heisenberg_multithreaded.jl`
- `examples/CSL.jl`

Run them only after the fixture scripts load, because the examples spawn worker
processes and are much heavier than a smoke test. For thesis regression, keep
both:

- an unmodified "reference full example" run to check upstream behavior,
- a shrunk smoke version with the same Hamiltonian construction but
  `sample_nr` and `maxiter` reduced, so it can run in CI or before cluster jobs.
