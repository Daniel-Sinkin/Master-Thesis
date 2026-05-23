# Regression And Invariant Testing Infrastructure

Testing should be deliberately overbuilt during the baseline/transpilation phase.
It is cheap to delete tests later; it is expensive to rediscover a sign,
index-order, or conjugation convention after optimization.

## Test Layers

1. C++ exact oracle tests:
   - product-state amplitudes,
   - local energy for simple Hamiltonians,
   - exact `O` rows by basis-tensor substitution,
   - parameter-space SR versus minSR on tiny systems,
   - weighted minSR,
   - compact sampled-sector Gram/direction versus dense sparse rows,
   - memory snapshot helper and byte estimators.

2. Julia reference fixtures:
   - export Julia outputs into JSON-lines,
   - import identical tensor/sample data into C++,
   - compare field-by-field with explicit tolerances,
   - store known Julia bugs/errors as fixture records rather than hiding them.
   - current C++ bridge: `real_3x2_D1_zero_sample`,
     `real_3x2_D2_zero_sample`, and `complex_3x2_D2_zero_sample` are embedded
     in the unit tests and check `logpsi`, `O_k`, `||O_k||^2`,
     Pauli-normalized Heisenberg energy, phase conventions, and the
     Julia-to-C++ tensor-order transpose for `D=2`; the `D=2` checks cover both
     all-zero and nonzero checker-pattern samples.
   - current Python bridge: the fixture validator reconstructs all small
     `theta_axis_labels` rows by explicit link enumeration and rechecks
     `logpsi`, `O_k` prefix, and `||O_k||^2` against Julia, including the
     complex `D=2` fixture. Additional `2x3,D=2` and `2x2,D=3` rows are kept
     for log-amplitude/boundary validation and to record the Julia two-row
     `E/O` indexing failure.

3. CUDA smoke tests:
   - projection kernels,
   - ragged physical-slice projection,
   - diagonal one-site/two-site energy,
   - dense Gram,
   - sampled-sector Gram,
   - sampled-sector `O^dagger x` scatter.

4. Precision regression:
   - FP64 CPU/CUDA baseline,
   - complex FP32 comparison,
   - TF32/cuBLASLt boundary bucket comparison,
   - optional FP16/BF16/Ozaki only after physics metrics exist.

5. Profiler regression:
   - stable Nsight Systems trace names,
   - CSV row per benchmark case,
   - fail a benchmark if dense `O` allocation exceeds target HBM fraction,
   - fail if unexpected host-device copies appear in the hot loop.

## Core Invariants

- `logpsi_exact` agrees with boundary `logpsi` within contraction cutoff.
- `local_energy_exact(S)` equals `sum_flips H[S,S'] psi(S') / psi(S)`.
- Dense `O` row has nonzeros only in sampled physical sectors.
- Compact sampled-sector Gram equals dense sparse-row Gram.
- Gram is Hermitian and diagonal entries are non-negative after ridge shift.
- Unit importance weights reproduce unweighted minSR.
- Normalized importance weights have mean one.
- Parameter-space SR equals sample-space minSR on tiny full-rank cases with the
  same ridge convention.
- `O^dagger x` scatter from compact rows matches dense scatter.
- Flip bucket classification is stable for diagonal, single-site, horizontal,
  vertical, plaquette, long-horizontal, and fallback supports.

## Fixture Matrix

Keep tiny fixtures fast and exhaustive:

| Name | Lattice | D | Type | Hamiltonian | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| product-heisenberg | `2x2` | 1 | real | Heisenberg | exact hand values |
| random-real-ragged | `3x2` | 2 | real | Heisenberg | open-boundary ragged dimensions |
| random-complex | `3x2` | 2 | complex | Heisenberg | conjugation/phase conventions |
| checker-real | `3x2` | 2 | real | Heisenberg | nonzero sample order and negative-amplitude log phase |
| checker-complex | `3x2` | 2 | complex | Heisenberg | nonzero sample order with complex phase |
| striped-real | `2x3` | 2 | real | Heisenberg | transposed lattice shape and two-row reference bug |
| striped-complex | `2x3` | 2 | complex | Heisenberg | transposed lattice shape with complex phase |
| checker-D3 | `2x2` | 3 | real | Heisenberg | higher virtual bond labels on a tiny lattice |
| diagonal-rydberg | `3x3` | 2 | real | Rydberg | diagonal long-range path |
| plaquette-csl | `3x3` | 2 | complex | CSL-style plaquette | four-body bucket |
| sampled-sector | `2x2` | 2 | complex | synthetic `E` | dense versus compact minSR |

## Memory Regression

Every executable that performs a full iteration should print:

- current RSS at start,
- after PEPS initialization,
- after Hamiltonian construction,
- after sampling,
- after `E/O`,
- after Gram/minSR,
- peak RSS.

On Linux cluster jobs, wrap with `/usr/bin/time -v`. On macOS reference checks,
wrap with `/usr/bin/time -l`. Keep both the process peak and internal allocator
facts where available (`Base.gc_live_bytes()` for Julia).

## CI-Like Local Commands

```bash
cmake --build code/peps_cuda/build
ctest --test-dir code/peps_cuda/build --output-on-failure
python3 -m py_compile code/peps_cuda/tools/estimate_peps_costs.py code/peps_cuda/tools/memory_pressure.py code/peps_cuda/tools/benchmark_matrix.py code/peps_cuda/tools/boundary_bucket_shapes.py code/peps_cuda/tools/occupancy_scratch.py
code/peps_cuda/tools/check_cuda_env.sh
code/peps_cuda/tools/run_cpu_regression.sh
```

`run_cpu_regression.sh` is the convenience wrapper for the local no-CUDA pack:
it configures/builds/tests FP64 and FP32 CPU variants, compiles the Python
validators, validates the Julia fixture file, and records the local CUDA/tooling
state.

Occasionally run a stricter compiler pass too:

```bash
cmake -S code/peps_cuda -B code/peps_cuda/build-warnings \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"
cmake --build code/peps_cuda/build-warnings
ctest --test-dir code/peps_cuda/build-warnings --output-on-failure
```

And a sanitizer pass for the CPU oracle:

```bash
cmake -S code/peps_cuda -B code/peps_cuda/build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build code/peps_cuda/build-asan
ctest --test-dir code/peps_cuda/build-asan --output-on-failure
```

Reference fixture export, slow but important:

```bash
code/peps_cuda/julia_reference/run_with_memory.sh \
  julia --project=code/peps_cuda/julia_reference --compiled-modules=no \
  code/peps_cuda/julia_reference/export_reference_fixtures.jl \
  code/peps_cuda/julia_reference/fixtures/reference_fixtures.jsonl
```

Validate the exported fixture file:

```bash
python3 code/peps_cuda/julia_reference/validate_reference_fixtures.py \
  code/peps_cuda/julia_reference/fixtures/reference_fixtures.jsonl
```

## Cluster Test Ladder

- A100 compile-only and smoke kernels.
- A100 one-GPU profile for `8x8,D=4,Ns=256`.
- A100 precision sweep FP64 versus FP32.
- GH200/JUPITER one-GPU repeat with SM90 build.
- JUPITER four-GPU sample sharding and Gram allreduce.
