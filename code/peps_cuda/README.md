# PEPS CUDA Scaffold

This is a standalone jump-off implementation for the finite-PEPS sampling and
TDVP/minSR pipeline from Puente, Weerda, Schroeder, and Rizzi. It deliberately
does not depend on `tensor-network/`; that code remains useful as learning and
visualization reference material only.

## What Is Implemented Now

- A small-system exact C++ CPU path:
  - PEPS tensors on an open rectangular lattice.
  - A physical-major packed tensor layout with per-site offsets for ragged
    open-boundary dimensions.
  - A site-wise parameter layout for future sliced/direct Gram accumulation.
  - Exact single-layer coefficient contraction `Psi(S)` by row dynamic
    programming over vertical boundary states.
  - Exact tiny-system sampling from `|Psi(S)|^2` by enumerating configurations.
    Batch generation caches the exact distribution once and draws from it, rather
    than enumerating once per sample.
  - Local-energy evaluation `E_loc(S) = <S|H|Psi>/Psi(S)` for generic local
    dense operator terms.
  - Hamiltonian helpers for nearest-neighbor Heisenberg, transverse-field Ising,
    and square-lattice Rydberg-style density interactions.
  - Flip contribution bucketing for diagonal, single-site, horizontal,
    vertical, plaquette, horizontal-long, and fallback energy paths.
  - Log-gradient rows `O_{S,i} = d Psi(S) / d theta_i / Psi(S)` by exact
    basis-tensor substitution. This is expensive but useful for debugging.
  - The minSR sample-space solve
    `theta_dot = -O^dagger (O O^dagger + lambda I)^-1 E`.
  - A tiny parameter-space SR solve
    `theta_dot = -(O^dagger O + lambda I)^-1 O^dagger E` for debugging
    and real-time/small-`Np` regimes.
  - A weighted minSR variant that scales rows by normalized importance weights.
  - A compact sampled-physical-sector minSR direction path that forms the
    sample-space Gram from compact rows and scatters the final update back into
    the dense parameter vector.
  - Normalized importance weights matching the Julia
    `compute_importance_weights` log-sum-exp formula.
- CUDA kernel stubs for the first GPU data-parallel pieces:
  - physical-slice projection for one sample or sample batches,
  - ragged batched projection using per-site slice sizes and offsets,
  - `|psi|^2` weights,
  - importance weights from `log(Psi)` and sampler probabilities.
  - diagonal nearest-neighbor Heisenberg energy accumulation.
  - generic one-site diagonal energy accumulation for fields/detunings.
  - generic two-site diagonal energy accumulation for long-range density terms.
  - dense sample-space minSR helpers for
    `T = O O^dagger + lambda I` and `-O^dagger x`.
  - weighted dense minSR helpers for importance-sampled rows.
  - sampled-physical-sector minSR Gram assembly that skips per-site dot
    products when two samples occupy different physical sectors.
  - sampled-physical-sector `-O^dagger x` scatter into the dense parameter
    update vector. For weighted minSR, pre-scale `x_s` by `sqrt(weight_s)`
    before this scatter.
- Unit tests for product-state amplitudes, local energy, gradient layout, flip
  classification, sampling, minSR output shape, and the Julia
  `real_3x2_D1_zero_sample`, `real_3x2_D2_zero_sample`, and
  `complex_3x2_D2_zero_sample` fixtures (`logpsi`, `O_k`, `||O_k||^2`,
  Pauli-normalized Heisenberg energy, and the Julia-to-C++ tensor-order
  transpose for `D=2`). The `D=2` checks also cover nonzero checker-pattern
  samples exported from Julia.

The CPU path is not meant for production sizes. It is the correctness mirror
for small lattices before replacing contractions with boundary-MPS/cuBLASLt
and CUDA kernels.

## Build Locally Without CUDA

```bash
cmake -S code/peps_cuda -B code/peps_cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build code/peps_cuda/build
ctest --test-dir code/peps_cuda/build --output-on-failure
./code/peps_cuda/build/peps_cuda_demo 2 2 2 4
./code/peps_cuda/build/peps_cuda_demo 2 2 2 3 tfi
./code/peps_cuda/build/peps_cuda_demo 2 2 2 3 rydberg
```

On a CUDA machine:

```bash
cmake -S code/peps_cuda -B code/peps_cuda/build-gpu -DCMAKE_BUILD_TYPE=Release -DPEPS_CUDA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build code/peps_cuda/build-gpu
ctest --test-dir code/peps_cuda/build-gpu --output-on-failure
./code/peps_cuda/build-gpu/peps_cuda_gpu_smoke
```

The CPU oracle defaults to complex FP64. For precision/memory experiments, build
the CPU path with complex FP32:

```bash
cmake -S code/peps_cuda -B code/peps_cuda/build-f32 -DCMAKE_BUILD_TYPE=Release -DPEPS_CUDA_REAL_TYPE=float
cmake --build code/peps_cuda/build-f32
ctest --test-dir code/peps_cuda/build-f32 --output-on-failure
```

## Scratch Tools

```bash
python3 code/peps_cuda/tools/estimate_peps_costs.py --gpu jupiter_gh200 --lx 16 --ly 16 --d 8 --dc 64 --samples 2000
python3 code/peps_cuda/tools/memory_pressure.py --lx 32 --ly 32 --d 8 --dc 128 --dc-double 16 --samples 5000 --hbm-gb 96
python3 code/peps_cuda/tools/benchmark_matrix.py --gpu jupiter_gh200
python3 code/peps_cuda/tools/boundary_bucket_shapes.py --lx 16 --ly 16 --d 8 --dc 64
python3 code/peps_cuda/tools/occupancy_scratch.py --arch h100 --threads 256 --regs 64 --smem-kib 48
code/peps_cuda/tools/check_cuda_env.sh
code/peps_cuda/tools/run_cpu_regression.sh
```

These are not profilers. They are quick sanity checks for dense-`O` memory,
boundary-MPS arithmetic scale, benchmark-case triage, approximate GEMM bucket
shapes, precision-dependent working-set pressure, and custom-kernel occupancy
ceilings before a real Nsight run. `run_cpu_regression.sh` runs the current
local no-CUDA regression pack: FP64 CPU build/tests, FP32 CPU build/tests,
Python validator syntax checks, Julia fixture validation, and the host
CUDA-environment probe.

## Julia Reference Fixtures

The reference harness is under `julia_reference/`. It exports JSON-lines
fixtures from the Julia/ITensor code, including tensor values, ITensor axis
labels, row-major samples for C++, boundary `logpsi`, `E_loc`, `O_k` prefixes,
and Julia memory facts.

Current fixture coverage:

- `real_3x2_D1_zero_sample`
- `real_3x2_D2_zero_sample`
- `real_3x2_D2_checker_sample`
- `complex_3x2_D2_zero_sample`
- `complex_3x2_D2_checker_sample`
- `real_2x3_D2_striped_sample`
- `complex_2x3_D2_striped_sample`
- `real_2x2_D3_checker_sample`

The Python validator reconstructs all small fixture rows by explicit link-label
enumeration. The C++ unit tests embed the `D=1`, real `D=2`, and complex `D=2`
fixture data and compare zero plus checker samples against Julia outputs.
The extra `2x3`/`D=3` rows currently validate `logpsi` and boundary
environments; their `E/O` fields intentionally record a Julia two-row
environment-indexing failure for later inspection.

## Cluster Templates

- `slurm/a100_profile.slurm`: build SM80 and profile `peps_cuda_gpu_smoke`.
- `slurm/jupiter_gh200_profile.slurm`: build SM90 for JUPITER Booster/GH200 and
  profile `peps_cuda_gpu_smoke`.

Related working notes live under `research/peps_cuda/`, especially
`boundary_mps_lowering.md`, `profiling_kpis.md`, and
`cluster_first_run_checklist.md`. `multi_gpu_strategy.md` sketches the first
JUPITER decomposition.

## Intended GPU Roadmap

1. Keep PEPS tensors in a padded site-major layout:
   `site -> physical -> north -> east -> south -> west`.
2. Generate or import Hamiltonian flip terms on CPU, bucketed by support shape:
   diagonal, horizontal 1/2-site, vertical 1/2-site, plaquette, and fallback.
3. Compute double-layer boundaries asynchronously and reuse them across many
   samples, as in Appendix B of the paper.
4. Use one stream per sample bucket, not one launch per contraction.
5. Lower boundary-MPS row absorption to grouped/strided GEMM calls first.
6. Replace hot small-GEMM buckets with cuBLASDx/CUTLASS/CuTe kernels only after
   Nsight Systems proves launch overhead or grouped-GEMM overhead dominates.
7. Build `O` row blocks and `E` on GPU, form `T = O O^dagger` with cuBLASLt,
   solve the small sample-space system, then compute `O^dagger x`.

The current CUDA kernels cover step 7 for the dense baseline. They are not
expected to beat cuBLASLt on the final cluster code, but they are useful for
checking data layout, stream plumbing, and Nsight visibility before the
boundary-MPS contractions are fully lowered.

The custom dense-Gram kernels are deliberately smoke-test scale. For real
`Ns=1000..5000` runs, benchmark cuBLAS/cuBLASLt Hermitian/GEMM paths and direct
sliced Gram accumulation before relying on one-block-per-Gram-entry kernels.

## Why The Exact CPU Path Is So Small

The production algorithm contracts PEPS approximately with boundary-MPS
compression. The exact CPU contraction here scales exponentially in row width
and is only for debugging shapes, signs, Hamiltonian term expansion, and the
O/E/minSR algebra on `2x2`, `3x3`, or similarly tiny cases.
