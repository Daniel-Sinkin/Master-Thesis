# PEPS CUDA Research Log

Timer:
- Start: 2026-05-14 23:13:21 CEST.
- Minimum checkpoint: 2026-05-15 01:13:21 CEST.
- Extended target after new reference-testing/memory/Ozaki scope:
  2026-05-15 02:00:00 CEST.

Scope:
- Target procedure: finite PEPS sampling, `E` and `O` generation, local-energy
  and log-gradient evaluation, and minSR/TDVP update.
- Implementation location: `code/peps_cuda`, separate from the learning
  `tensor-network` library.

## Source Bundle

Stored under `research/peps_cuda/sources/`:

- `puente_weerder_schroeder_rizzi_2025_finite_peps.pdf`: arXiv 2503.12557.
- `puente_weerder_schroeder_rizzi_2025_finite_peps_source.tar.gz` plus extracted
  `main.tex`, images, and bibliography.
- `PEPS_structure_documentation.pdf`: local 12-page theory/workflow document.
- `QuantumNaturalfPEPS.jl-main.zip` plus extracted source snapshot.
- `vieijra_haegeman_verstraete_vanderstraeten_2021_direct_peps_sampling.pdf`.
- `chen_heyl_2023_minsr_neural_quantum_states.pdf`.
- `wu_nys_2026_peps_tvmc_gpu.pdf`.
- `chen_jiang_hangleiter_schuch_2025_sign_problem_tn_contraction.pdf`.
- `hopper_tuning_guide.pdf`.
- `cuda_c_best_practices_guide.pdf`.
- `nsight_compute_profiling_guide.pdf`.
- `nvidia_a100_datasheet.pdf`.
- `nvidia_h100_datasheet.pdf`.
- `nvidia_h200_datasheet.pdf`.
- `menczer_legeza_2024_dgxh100_dmrg.pdf`.
- `menczer_legeza_2023_hybrid_cpu_gpu_tns.pdf`.
- `menczer_legeza_2023_nonabelian_tns_gpu.pdf`.
- `gh200_data_movement_2408.11556.pdf`.
- `brower_legeza_2025_blackwell_fp64_emulation.pdf`.
- `grace_hopper_blas_offload_2404.13195.pdf`.
- `annotated_sources.md` gives per-source viability and implementation notes.
- `performance_plan.md` gives the staged profiling/optimization plan.
- `hardware_notes.md` gives A100/H100/H200/JUPITER hardware facts and PEPS
  memory implications.
- `implementation_map.md` maps the Julia reference stages to the current
  C++/CUDA scaffold and remaining production pieces.
- `memory_hierarchy_notes.md` records HBM/L2/shared/register placement guidance
  for the PEPS hot loop.
- `boundary_mps_lowering.md`, `profiling_kpis.md`,
  `reference_algorithm_anchors.md`, `tensor_network_gpu_sota.md`, and
  `cublas_grouped_gemm_plan.md` collect the current implementation/profiling
  synthesis.
- `source_inventory.md` lists the local source bundle and external links.
- `references.bib` contains BibTeX entries for core PEPS, direct sampling,
  minSR, tensor-network GPU, and GH200 data movement sources.

## Paper: Puente, Weerda, Schroeder, Rizzi 2025

Link: https://arxiv.org/abs/2503.12557

Viability:
- This is the primary algorithmic source. It explicitly states the finite-PEPS
  sampling pipeline and maps directly onto `QuantumNaturalfPEPS.jl`.
- It is recent and exactly aligned with the thesis target.

Key learnings:
- The expectation estimator is
  `<O> = sum_S |Psi(S)|^2/<Psi|Psi> * <S|O|Psi>/Psi(S)`.
- For optimization, the paper avoids the parameter-space matrix
  `G = O^dag O` of size `Np x Np` and uses minSR:
  `theta_dot = O^dag (O O^dag)^-1 E_loc`, where `O O^dag` is only
  `Ns x Ns`.
- For finite PEPS, `Np = Lx * Ly * D^4 * d`; with `D=8`, this is huge even
  before complex storage, while `Ns` is often 1000-5000 in imaginary time.
- Boundary-MPS is the core contraction primitive. Paper scaling:
  single layer: `O(Dc^3 D^3) + O(Dc^2 D^4)`;
  double layer: `O(Dc^3 D^4) + O(d Dc^2 D^6)`.
- Local Hamiltonian terms are efficient because flipped configurations differ
  only on small support. Horizontal/vertical/plaquette buckets matter.
- Diagonal long-range interactions are cheap in the sampling estimator because
  `O_SS Psi(S)` needs no extra contraction.
- Direct sampling needs double-layer boundaries, but the paper argues these can
  use small double-layer bond dimension and be reused/asynchronously refreshed.
- Appendix B traces direct sampling to conditional row reduced density matrices,
  with a lower double-layer boundary `D^l`, sampled upper boundary `E^u`, and
  importance correction for the approximate sampling probability.

CUDA implications:
- The GPU implementation should optimize row absorption and contraction reuse,
  not the small minSR solve first.
- `O` and `E` generation are embarrassingly parallel across samples once the
  current PEPS/double-layer environments exist.
- Horizontal/vertical/plaquette flip classes should be separate kernels or
  separate GEMM buckets.

## PEPS Structure Documentation

Local source: `research/peps_cuda/sources/PEPS_structure_documentation.pdf`.

Viability:
- This is the best compact "what code must do" source. It is clearer than the
  paper for implementation ordering and typical problem sizes.

Key learnings:
- It states the computational tasks plainly:
  evaluate `Psi_S`, evaluate derivatives `d Psi_S / d theta_i`, solve for
  `theta_dot`, integrate, generate samples and probabilities.
- Typical imaginary-time sizes: `8x8` to `32x32`, bond dimensions `D=2..8`,
  local dimension usually `d=2`, sometimes up to `d=8`, samples about
  `1000-5000`.
- Real-time is harder: smaller lattices around `8x8`, far more samples, and
  often `Ns ~ Np`.
- Double-layer environments dominate later imaginary-time runs at large `D`.
- The direct-sampling algorithm is:
  `E^u[0]=1; for rows i: T^u_{S<i}[i]=mul(T[i,:],E^u[i-1];Ds); sample row;
  project row; E^u[i]=mul(E^u[i-1],T_proj[i];Dc)`.

CUDA implications:
- A production CUDA code should expose `Dc`, `Ds`, and double-layer `Dc_double`
  separately.
- Async stale double-layer boundaries are a legitimate approximation to explore;
  importance weights and energy error need to be monitored.
- The fastest path probably uses a persistent pool of sample workers per GPU,
  all reusing the same PEPS tensors and environment buffers.

## QuantumNaturalfPEPS.jl Snapshot

Link: https://github.com/KonradSchroeder/QuantumNaturalfPEPS.jl

Viability:
- This is the reference implementation to mirror semantically, not to optimize
  mechanically.

Important files:
- `src/sampling.jl`: double-layer environments, direct sampling row logic.
- `src/Environments.jl`: `get_logpsi_and_envs`, vertical and horizontal
  environment construction, normalization factors.
- `src/Ek.jl`: sorting flip terms into horizontal, vertical/four-body, longer
  horizontal, and fallback; local energy via flipped amplitudes.
- `src/Ok.jl`: log-gradient contraction with one tensor removed.
- `src/Distributed/Oks_and_Eks*.jl`: parallel sample-loop structure.

CUDA implications:
- The Julia code parallelizes samples across CPU threads/processes, but the hot
  contractions still resolve through ITensor abstractions. GPU code should
  make tensor layouts and contraction buckets explicit.
- The function-level equivalent to target is:
  `get_sample -> get_logpsi_and_envs -> get_all_horizontal_envs -> get_Ek/get_Ok`.

## JUPITER / GH200 / H100-H200 Hardware

Primary links:
- JUPITER GPU docs: https://apps.fz-juelich.de/jsc/hps/jupiter/gpu-computing.html
- JUPITER configuration: https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html
- NVIDIA H200 specs: https://www.nvidia.com/en-gb/data-center/h200/
- Hopper tuning guide: https://docs.nvidia.com/cuda/archive/12.1.0/hopper-tuning-guide/index.html

Facts to use:
- JUPITER Booster nodes have 4 NVIDIA GH200 Grace-Hopper Superchips.
- Each JUPITER Booster GH200 includes a Hopper GPU with 132 SMs, 96 GB HBM3,
  and about 4 TB/s HBM bandwidth.
- Each GH200 has one Grace CPU with 72 Arm Neoverse-V2 cores, 120 GB LPDDR5X
  memory at 512 GB/s on standard compute nodes.
- CPU-GPU NVLink-C2C is listed at 900 GB/s.
- GPU-to-GPU NVLink 4 links are listed as 300 GB/s between pairs
  (150 GB/s per direction).
- JUPITER Booster nodes have 4 x InfiniBand NDR200 ConnectX-7.
- JSC notes that Nsight Compute may lock clocks to base frequency by default;
  this matters for comparing clock-sensitive metrics.
- H200 SXM has 141 GB HBM3e and 4.8 TB/s bandwidth, but current JUPITER docs
  say the Booster GPU is H100/GH200 with 96 GB HBM3, not H200.

Optimization consequences:
- Treat JUPITER as GH200/H100-class SM90 with more coherent CPU-GPU bandwidth
  than a PCIe host, but keep hot sample loops on HBM.
- Do not assume unified memory placement is free. Explicit HBM residency still
  matters.
- One MPI rank per GPU is the default JSC affinity story; later multi-GPU PEPS
  should shard samples or Hamiltonian buckets across the four GPUs.

## CUDA / Hopper Guidelines

Sources:
- CUDA C++ Best Practices Guide.
- Hopper Tuning Guide.
- Nsight Compute Profiling Guide.
- Simon Boehm CUDA matmul worklog: https://siboehm.com/articles/22/CUDA-MMM
- cuBLAS grouped GEMM blog: https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
- cuTENSOR/cuTensorNet docs for tensor contraction alternatives.

Key learnings:
- First priorities remain: expose parallelism, minimize host-device transfer,
  coalesce global memory, reduce redundant global memory, avoid warp divergence.
- Hopper has 64 resident warps/SM like Ampere, 64K 32-bit registers/SM, up to
  255 registers/thread, 228 KB shared memory/SM, and 227 KB max shared memory
  per block.
- Hopper TMA can move 1D-5D tensors between global and shared memory with less
  register pressure and supports warp-specialized loading.
- H100 increases L2 to 50 MB and supports L2 persistence controls.
- Boehm's worklog is still the right mental model: coalescing, shared-memory
  tiling, arithmetic-intensity via register tiling, vectorized loads, autotuning,
  and warp tiling. For PEPS, use libraries for GEMM first; write custom kernels
  only where grouped/batched GEMM loses on irregular tiny shapes or launch count.
- cuBLAS 12.5 grouped GEMM supports different matrix sizes/transposes/scales in
  one launch and is explicitly meant to reduce loops over many small matmuls.
- cuBLASLt heuristic tuning is required on Ampere+ for serious benchmarking.
- cuTENSOR should be benchmarked for dense tensor contractions that otherwise
  require explicit transposes; cuTensorNet is more of a baseline/path-planning
  tool than a natural replacement for boundary-MPS sampling loops.
- CUDA Graphs are a likely second-pass fix for repeated small-kernel sample-loop
  topology; cuBLASDx/CuTe should wait until cuBLASLt/grouped GEMM/cuTENSOR
  baselines identify fixed tiny contractions that are still bottlenecks.

Profiler KPI pack:
- Nsight Systems first:
  kernel launch count, CPU gaps, CUDA API time, memcpy/UM migrations, stream
  overlap, MPI/NVLink/IB overlap later.
- Nsight Compute second:
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
  `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`,
  `l1tex__t_sector_hit_rate.pct`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`,
  `smsp__warps_eligible.avg.per_cycle_active`,
  `smsp__issue_active.avg.pct_of_peak_sustained_active`,
  branch/warp execution efficiency metrics for flip-term kernels,
  tensor-core pipe active metrics for GEMM/cutensor/cublasLt paths.

## Tensor-Network GPU SOTA

Sources:
- Menczer et al., DGX-H100 DMRG, arXiv:2407.07411.
- Menczer and Legeza, Massively Parallel Tensor Network State Algorithms on
  Hybrid CPU-GPU Architectures, JCTC 2025.
- Menczer and Legeza, Tensor Network State Algorithms on AI Accelerators, JCTC
  2024.
- Brower et al., Blackwell FP64 emulation for DMRG, arXiv:2510.04795.
- Grace-Hopper BLAS offload / data movement papers for CPU-GPU placement
  intuition on GH200-class nodes.

Viability:
- These are not finite-PEPS sampling codes, but they are highly relevant for
  scheduling, GPU task grouping, data reuse, and tensor-core precision strategy.

Key learnings:
- The H100 DMRG paper reports 246 TFLOP/s sustained on a DGX-H100 node and
  emphasizes hybrid CPU-multiGPU execution.
- The JCTC hybrid CPU-GPU work frames tensor-network workloads as many
  independently executable vector/matrix tasks with careful scheduling and data
  dependency reuse.
- The useful transfer to PEPS is task bucketing by shape and data locality:
  keep contractions requiring the same tensor/environment blocks adjacent.
- Blackwell FP64-emulation work is relevant as a future precision idea, but this
  thesis target is A100 then GH200/H100/H200, so it should not affect kernels
  except as a caution to isolate precision policy.
- GH200 data-movement papers are relevant mainly for host/device placement:
  NVLink-C2C makes CPU-GPU collaboration less awful than PCIe, but PEPS hot
  tensors/environments still need to live in HBM for the sample loop.

## Initial Performance Hypothesis

Likely hotspots before profiling:
- Double-layer boundary refresh at large `D`.
- Single-layer boundary contractions for each sample.
- Horizontal/four-body flipped-amplitude evaluation for `E_loc`.
- Log-gradient `O` construction, because each sample produces a huge sparse row.
- Launch overhead if every contraction maps to a separate small GEMM/kernel.
- Memory traffic and allocation churn if environments are rebuilt per sample
  without pooling.

First optimization passes:
- Baseline CPU exact for tiny tests; then CUDA naive shape-mapped kernels.
- Move to cuBLASLt grouped GEMM for row-absorption buckets.
- Pool all device buffers; no allocation in the sample loop.
- Use one stream per bucket/sample group, not one stream per tiny contraction.
- Add NVTX ranges named after Julia stages:
  `double_layer_envs`, `sampling`, `vertical_envs`, `horizontal_envs`,
  `energy`, `log_gradients`, `minsR`.
- Store `O` in sample-major order initially for `T=O O^dag`, then test a
  parameter-major transpose if `O^dag x` dominates.

## Implementation Checkpoint: 2026-05-14 23:23 CEST

Files added:
- `code/peps_cuda/include/peps_cuda/peps.hpp`
- `code/peps_cuda/src/peps_cpu.cpp`
- `code/peps_cuda/src/peps_cuda.cu`
- `code/peps_cuda/src/main.cpp`
- `code/peps_cuda/tools/estimate_peps_costs.py`
- `code/peps_cuda/slurm/jupiter_gh200_profile.slurm`

Current status:
- CPU exact scaffold configures and builds locally with AppleClang.
- Demo runs for `2x2 D=2` and `3x3 D=2`.
- CUDA code is intentionally gated by `PEPS_CUDA_ENABLE_CUDA`; local machine has
  no `nvcc`, so CUDA compile is deferred to the cluster.
- Added an explicit flip-contribution classifier mirroring the Julia `sort_dict`
  idea. It produces buckets for diagonal, single-site, horizontal nearest,
  vertical nearest, plaquette, horizontal-long, and fallback terms.

Known limitations:
- CPU sampler enumerates the whole Hilbert space and is therefore only for tiny
  tests.
- CPU gradients are exact but computed by basis-tensor substitution, so they are
  intentionally slow.
- Boundary-MPS compression is not implemented yet. The next real implementation
  step is replacing exact contraction with row-MPS absorption and truncation
  backed by grouped GEMM/SVD routines.
- The current CUDA file has only the first simple data-parallel kernels; the
  production contraction kernels still need to be written after the memory layout
  is frozen.

## Implementation Checkpoint: 2026-05-14 23:30 CEST

Additional files:
- `code/peps_cuda/include/peps_cuda/cuda_kernels.hpp`
- `code/peps_cuda/src/tests.cpp`
- `code/peps_cuda/src/gpu_smoke.cu`
- `code/peps_cuda/tools/occupancy_scratch.py`

CUDA scaffold expanded:
- Added batch physical-slice projection with sample-major spins and output.
- Added diagonal Heisenberg energy accumulation for the cheap `S^z S^z` part of
  nearest-neighbor models.
- Added dense minSR kernels for `T = O O^dagger + lambda I` and
  `theta_dot = -O^dagger x`. This is the simple baseline before switching to
  cuBLASLt/cuSOLVER for the production path.
- Added C++ normalized importance weights matching the Julia formula
  `exp(log_ratio - (logsumexp(log_ratio)-log(Ns)))`; CUDA weight kernel now
  accepts the same normalization constant instead of producing raw ratios.
- Added weighted CPU minSR by scaling sample rows with `sqrt(weight)`, matching
  the paper's sampled weighted least-squares formulation.
- Added weighted CUDA dense minSR helper kernels as the first GPU baseline for
  importance-sampled `O` rows.
- Added a CUDA smoke-test executable that launches the current kernels and
  checks tiny outputs. The JUPITER Slurm profile script now points Nsight
  Systems/Compute at this GPU smoke test instead of the CPU-only demo.
- Added an A100 Slurm profiling template that builds SM80 and profiles the same
  CUDA smoke test.
- CUDA CMake target now respects `CMAKE_CUDA_ARCHITECTURES`, so A100 and
  JUPITER scripts can build only SM80 or SM90 instead of always compiling both.
- Adjusted the JUPITER Slurm script to let Slurm manage `CUDA_VISIBLE_DEVICES`
  per task and to pass `--cpus-per-task` to `srun`, matching the JUPITER GPU and
  affinity documentation.
- Added a host packed PEPS layout with physical-major site slices, per-site
  offsets, projected-output offsets, and alignment. Added a ragged CUDA
  projection kernel so open-boundary sites do not have to be padded to the
  largest interior slice for correctness.
- Added a site-wise parameter layout helper for future sparse/sliced `O` and
  direct Gram accumulation.
- Added Hamiltonian builders for transverse-field Ising and square-lattice
  Rydberg-style long-range density interactions, so the scaffold can exercise
  single-site flip buckets and long-range diagonal `E` terms beyond Heisenberg.
- Demo now accepts `heisenberg`, `tfi`, or `rydberg` model names. Local runs for
  `2x2 D=2` show TFI/Rydberg single-site flip buckets as expected.
- Added a generic CUDA two-site diagonal energy kernel for long-range
  density-density Hamiltonian terms, separate from the Heisenberg-specific
  `S^z S^z` helper.
- Added a generic CUDA one-site diagonal energy kernel for detuning/Z-field
  terms.
- Removed `--use_fast_math` from the CUDA compile flags because FP64 correctness
  should be the default baseline.

Verification:
- `cmake --build code/peps_cuda/build` succeeds locally.
- `ctest --test-dir code/peps_cuda/build --output-on-failure` passes 5/5 tests.
- CUDA target is still not compiled locally because `nvcc` is unavailable.
- `which nvcc` returns `nvcc not found` on this machine.
- Demo now prints dense-`O` byte estimates and sample-major spin buffer size, so
  the CPU reference path exposes the first CUDA transfer/layout quantities.
- Demo now prints Hamiltonian flip-bucket counts for the first sample, making
  the E-side split visible before writing specialized bucket kernels.
- CPU exact batch sampling now builds the exact tiny-system distribution once
  per batch and draws from the cumulative weights, instead of re-enumerating for
  every requested sample.
- Occupancy scratch examples:
  `256 threads, 64 regs/thread, 48 KiB smem` gives a theoretical ceiling of
  50% occupancy on H100/GH200 but 37.5% on A100 because shared memory limits the
  latter to three resident blocks/SM. A `128 thread, 96 regs/thread, 96 KiB smem`
  custom kernel would be only 12.5% on H100/GH200, so such a design needs high
  arithmetic intensity to make sense.

Cost-model snapshots for `16x16, D=8, Dc=64, Ns=2000`:
- Approximate bulk parameter count is 2,097,152, so dense `O` would occupy
  about 62.5 GiB; sampled-physical-sector `O` is still about 31.25 GiB, while
  the `Ns x Ns` Gram is only about 61 MiB.
- A100 SXM 40GB: single-layer sample contractions have a compute-bound lower
  bound around 7.47 s at native FP64 cores.
- H100/H200/JUPITER GH200-class: same arithmetic gives about 2.13 s at native
  FP64 cores and 1.08 s at FP64 tensor-core peak, before launch/truncation/memory
  realities.
- JUPITER GH200 `32x32, D=8, Dc=96, Ns=5000`: naive single-layer work estimate
  is about 2434 TFLOP-ish, lower bound about 71.6 s at FP64 core peak; dense
  `O` would be about 625 GiB, sampled-sector `O` about 312.5 GiB, while the Gram
  is about 381 MiB. This makes boundary reuse and direct/sliced Gram accumulation
  non-negotiable.

Bond-dimension sweep for `16x16, Ns=2000` on GH200 assumptions:
- `D=4, Dc=32`: dense `O` 3.91 GiB, all-sample single-layer work
  1.13 TFLOP-ish, ideal FP64 lower bound 0.033 s.
- `D=6, Dc=48`: dense `O` 19.78 GiB, work 12.90 TFLOP-ish, lower bound 0.379 s.
- `D=8, Dc=64`: dense `O` 62.50 GiB, work 72.48 TFLOP-ish, lower bound 2.13 s.
The simple model makes clear that launch overhead and bad contraction lowering
will dominate at small `D`, while memory layout/Gram strategy and boundary reuse
become unavoidable at `D=8`.

## Implementation Checkpoint: 2026-05-15 00:05 CEST

Added a cluster-first-run layer:
- `code/peps_cuda/tools/benchmark_matrix.py` emits CSV benchmark cases with
  dense-`O`, sampled-sector-`O`, Gram storage, and a rough memory triage label.
- `research/peps_cuda/cluster_first_run_checklist.md` records the first A100 and
  GH200 smoke/profiling sequence, including Nsight Systems and Nsight Compute
  commands and the initial bad-trace signatures to watch for.
- `code/peps_cuda/README.md` now points at the benchmark matrix tool alongside
  the cost and occupancy scratch tools.

Fresh benchmark-matrix sanity check for JUPITER GH200 assumptions:
- `16x16, D=8, Ns=2000`: dense `O` is about 62.5 GiB and sampled-sector `O`
  about 31.25 GiB, so dense `O` is a plausible debug baseline only if the rest
  of the run is kept small.
- `16x16, D=8, Ns=5000`: dense `O` is about 156.25 GiB and sampled-sector `O`
  about 78.125 GiB, so direct/sliced Gram should be used.
- `32x32, D=8, Ns=2000`: dense `O` is about 250 GiB and sampled-sector `O`
  about 125 GiB, so direct/sliced Gram is already mandatory.
- `32x32, D=4, Ns=5000`: dense `O` is about 39.06 GiB, so it is a reasonable
  stress/debug case on 96 GB GH200 before moving to `D=8`.

Added sampled-sector `O`/Gram support:
- Host API now exposes `sampled_sector_parameter_count`,
  `sampled_sector_o_bytes`, `compact_sampled_sector_log_gradients`, and
  `sampled_sector_gram`.
- Added `minsr_direction_sampled_sector` and weighted variant. These solve the
  sample-space minSR system from compact rows, then scatter the final update
  back into the dense parameter vector. Unit tests compare this against dense
  minSR on sparse rows.
- The compact row stores only one physical slice per site. When forming
  `T_ss'`, the dot contribution for a site is skipped if sample `s` and `s'`
  used different physical values there. This exactly mirrors the sparsity of the
  dense `O` row without storing the zero physical sectors.
- CUDA now has `launch_sampled_sector_minsr_gram`, a smoke-test-scale kernel
  for the same compact representation. It is not the final production strategy,
  but it validates the indexing/dataflow that the later direct Gram will reuse.
- CUDA now also has `launch_sampled_sector_minsr_apply_odag`, which scatters the
  solved sample-space vector back into the dense parameter update using atomic
  adds over the selected physical sectors.
- CPU unit tests now check compact sampled-sector rows and a Gram entry where
  one site is skipped because the two samples differ.
- GPU smoke test now includes the sampled-sector Gram and sampled-sector
  `O^dagger` scatter kernels. This cannot be compiled locally without `nvcc`,
  but the CUDA launch surface and expected values are in place for the first
  cluster run.

Verification after this change:
- `cmake --build code/peps_cuda/build` succeeds.
- `ctest --test-dir code/peps_cuda/build --output-on-failure` passes 5/5 tests.

Added `research/peps_cuda/tensor_network_gpu_sota.md`:
- Synthesis of PEPS-specific sources, Menczer/Legeza GPU tensor-network work,
  NVIDIA library baselines, GH200 implications, and what those imply for
  sampling, `E`, `O`, precision, and profiler order.
- The main implementation takeaway is that the first serious optimization pass
  should benchmark grouped GEMM/cuBLASLt/cuTENSOR by PEPS shape bucket before
  writing bespoke kernels.

Added `research/peps_cuda/boundary_mps_lowering.md`:
- Explicit index formulas for single-layer and double-layer row absorption.
- GEMM reshapes for boundary absorption buckets.
- Notes on compression API boundaries, horizontal environments, compact
  sampled-sector `O`, Hamiltonian `E` buckets, and first synthetic shape buckets
  to benchmark.

Added `code/peps_cuda/tools/boundary_bucket_shapes.py`:
- Emits CSV GEMM-shape buckets from the boundary-MPS lowering formulas.
- Example for `16x16, D=8, Dc=64` shows the dominant synthetic bucket as
  `single_absorb M=4096,N=512,K=8,count=196` and
  `double_absorb M=4096,N=262144,K=64,count=196` under the simple compressed-chi
  model. These are intentionally approximate, but they give the first grouped
  GEMM/cuTENSOR benchmark manifest.
- `python3 -m py_compile` passes for all current Python scratch tools.

Added `research/peps_cuda/profiling_kpis.md`:
- Concrete Nsight Systems questions, Nsight Compute metrics by kernel family,
  NVTX range naming, roofline sanity formulas, CSV schema, and stop conditions
  for optimization passes.
- Key warning: dense `O` should be abandoned as soon as it is no longer a
  correctness/debugging baseline, because optimizing an impossible memory object
  is wasted effort.

Added `research/peps_cuda/reference_algorithm_anchors.md`:
- Condensed implementation anchors from the paper and Julia source: minSR
  sample-space solve, boundary-MPS costs, Appendix-B direct sampling, stale
  double-layer environment logic, diagonal long-range term handling, Julia
  `Ok/Ek` stage order, and the exact importance-weight formula.

Housekeeping:
- Extended `code/peps_cuda/.gitignore` to avoid accidentally committing local
  build/profile outputs from CMake, Nsight Systems, Nsight Compute, and Slurm
  smoke runs.
- Added `research/peps_cuda/source_inventory.md` with the local PDF/archive
  bundle, approximate sizes, and external links to recheck before thesis
  citation.
- Downloaded two additional useful papers into the source bundle:
  `wu_nys_2026_peps_tvmc_gpu.pdf` and
  `chen_jiang_hangleiter_schuch_2025_sign_problem_tn_contraction.pdf`.
- Added annotated-source notes and BibTeX entries for both. The Wu/Nys paper is
  especially relevant because its "small-o trick" matches the compact
  sampled-sector `O` strategy now implemented here.
- Added `research/peps_cuda/cublas_grouped_gemm_plan.md`, mapping boundary-MPS
  shape buckets to looped cuBLAS, strided batched GEMM, grouped GEMM, cuBLASLt,
  and cuTENSOR benchmark baselines.
- Extended `estimate_peps_costs.py` with sampled-sector/direct-Gram dot element
  estimates. For `32x32, D=8, d=2`, a random-sample direct compact Gram uses an
  estimated 2,097,152 dot elements per sample pair versus 8,388,608 for dense
  materialized `O`, a `0.25` ratio before layout/reuse effects.
- Added a CPU unit test that weighted sampled-sector minSR matches dense
  weighted minSR on sparse rows, covering the importance-weighted path.
- Rechecked the paper/Jl source for centering conventions: the main finite-PEPS
  derivation uses raw `O = dPsi/Psi` and `E_loc = <S|H|Psi>/Psi(S)` rows in the
  sampled least-squares/minSR equation, with a note that explicit normalization
  would add another term. The scaffold keeps this raw convention and documents
  centered/gauge-fixed tVMC variants as optional future solver variants.
- Re-read the local PEPS structure PDF. It matches the task split and workload
  assumptions already recorded (`8x8..32x32`, `D=2..8`, `Ns~1000..5000` for
  imaginary time; smaller lattices with much larger `Ns` for real time). Noted a
  likely equation-number typo: Sec. 3.3 says use Eq. 12 when `Ns << Np`, but
  Eq. 12 is the parameter-space inverse in that document; the text clearly means
  the sample-space/minSR equation Eq. 13.
- Added `sr_direction_parameter_space` as a tiny CPU/debug solve for the
  parameter-space form. Unit tests confirm it matches minSR under the same ridge
  regularization on small sparse rows.
- Added the Wu/Nys small-o formula to `reference_algorithm_anchors.md`, matching
  their masked compact-sector construction to the scaffold's
  `sampled_sector_gram` and CUDA sampled-sector Gram kernel.
- Added `code/peps_cuda/tools/check_cuda_env.sh`, a cluster sanity script that
  records host/module/compiler/NVIDIA/Slurm context and prints the default PEPS
  cost and boundary-bucket estimates. It runs locally too, reporting the expected
  missing `module`, `nvcc`, and `nvidia-smi` on this Mac.
- Added `research/peps_cuda/multi_gpu_strategy.md`: one MPI rank per GPU,
  sample sharding first, Gram/allreduce options, compact `O^dagger x` update
  reduction, and when to consider finer-grained contraction sharding.
- Updated `memory_hierarchy_notes.md` so direct Gram accumulation explicitly
  skips site contributions when two samples selected different physical sectors,
  and notes that the current scaffold is sample-major while a production kernel
  may transpose to site-major blocks.
- Added `research/peps_cuda/README.md` as an index for the research bundle.
- Added `research/peps_cuda/next_implementation_steps.md`, a prioritized
  continuation plan from A100 smoke compile through NVTX, synthetic GEMM
  benchmarks, boundary-MPS data structures, compact `O`, direct sampling, and
  multi-GPU sample sharding.

## Implementation Checkpoint: 2026-05-15 01:20 CEST

Reference alignment and Julia environment:
- Added `code/peps_cuda/julia_reference/` as a deliberately isolated Julia
  harness. It uses a local `ParallelGradient` stub and a local snapshot of
  `QuantumNaturalGradient.jl`, because the upstream reference stack depends on
  unregistered packages and ships no reproducible Manifest.
- Julia 1.11.5 can run the fixture exporter when invoked with
  `--compiled-modules=no`. This avoids a `QuantumNaturalGradient.__init__`
  precompile bug where `pathof(...)` can be `nothing` and is passed to
  `occursin`.
- The harness pins older ITensors/ITensorMPS versions and adds small
  compatibility shims for symbols the reference source expects.
- Exported `code/peps_cuda/julia_reference/fixtures/reference_fixtures.jsonl`
  with four rows: metadata plus real `D=1`, real `D=2`, and complex `D=2`
  `3x2` zero-sample cases.
- Validation command:
  `python3 code/peps_cuda/julia_reference/validate_reference_fixtures.py code/peps_cuda/julia_reference/fixtures/reference_fixtures.jsonl`
  reports `rows=4 logpsi_rows=3 boundary_errors=0 max_env_error=8.882e-16`.
- The latest successful Julia fixture export, wrapped with `/usr/bin/time -l`,
  took about 116 s on this Mac and reported maximum resident set size
  `1665908736` bytes, peak memory footprint `1478821712`, and fixture-level
  `Base.gc_live_bytes()` values around 291-424 MB.
- Added a C++ unit-test bridge for the Julia `real_3x2_D1_zero_sample` fixture.
  It checks `logpsi`, the first `O_k` entries, `||O_k||^2`, and the
  Pauli-normalized Heisenberg energy. The energy check uses `J=4` in the C++
  helper because the Julia fixture uses ITensor Pauli `X,Y,Z`, while the C++
  helper is spin-operator normalized.

Julia code critique highlights:
- Good: clear stage order (`sample -> logpsi/envs -> horizontal envs -> E/O`),
  explicit flip-term sorting, compact physical-sector structure in `Ok.jl`,
  log-sum-exp importance weights, and the minSR sample-space solve when
  `Ns < Np`.
- High-ROI problems: dense `Oks = Matrix(length(peps), sample_nr)` storage,
  repeated allocation/ITensor abstraction in the sample loop, dynamic
  dictionaries/tuple keys for flip buckets, no explicit CUDA residency path,
  and no manifest/test fixtures in the upstream repo.
- Likely reference issues to preserve/document rather than silently "fix" in
  the first transpilation: `get_logψ_and_envs` default `pos=length(env_top)÷2`
  breaks two-row systems, `Ek.jl` has a `vetr` typo on the vertical path, and
  `NaturalGradient.jl` appears to compute `Eks_eff` then overwrite it before
  `tdvp_error`.

Precision and memory-pressure snapshots:
- `2x2,D=2,Ns=4` CPU demo:
  FP64 dense `O` = 2048 B, sampled-sector `O` = 1024 B, peak RSS about
  1.39 MiB. FP32 dense `O` = 1024 B, sampled-sector `O` = 512 B, peak RSS about
  1.38 MiB. The tiny process RSS is dominated by executable/runtime overhead,
  but the explicit `O` byte counters halve as expected.
- `16x16,D=8,Ns=2000,HBM=96 GiB`:
  FP64 dense total about 62.62 GiB (65.2% HBM), sampled-sector total about
  31.37 GiB (32.7% HBM). FP32 dense total about 31.31 GiB (32.6% HBM),
  sampled-sector total about 15.68 GiB (16.3% HBM).
- `32x32,D=8,Ns=5000,HBM=96 GiB`:
  FP64 dense total about 625.56 GiB and sampled-sector total about 313.06 GiB;
  FP32 dense total about 312.79 GiB and sampled-sector total about 156.54 GiB.
  Even compact FP32 exceeds one 96 GiB GH200, so this regime needs direct
  streamed Gram accumulation and/or multi-GPU sample sharding.
- H100/GH200 occupancy scratch for `256 threads, 64 regs/thread, 32 KiB smem`
  gives 4 active blocks/SM and 50% theoretical occupancy. A100 gives the same
  for this particular point, but shared memory becomes the tighter limiter as
  the per-block scratch grows.

Ozaki/Ozaki-II conclusion:
- Downloaded `ozaki_scheme_ii_2504.08009.pdf` and
  `mukunoki_2025_dgemm_without_fp64_ozaki_fp8.pdf`.
- Ozaki-II is technically interesting for FP64 emulation on tensor cores and
  future Blackwell-style weak-FP64 regimes, but it should not be first-pass work
  for this PEPS baseline on A100/H100/GH200. The initial bottlenecks are likely
  boundary-MPS lowering, launch count, irregular grouped contractions,
  allocation/data movement, and dense-`O` memory pressure, not one huge
  compute-bound GEMM that needs FP64 emulation.
- Keep precision policy isolated so Ozaki-II can be a later backend for large
  compute-bound GEMM buckets if profiler data says native FP64 tensor cores or
  FP32/TF32 physics accuracy are not enough.

Fresh verification:
- `clang-format` applied to changed C++ files.
- `cmake --build code/peps_cuda/build` succeeds.
- `ctest --test-dir code/peps_cuda/build --output-on-failure` passes 5/5.
- `cmake --build code/peps_cuda/build-f32` succeeds.
- `ctest --test-dir code/peps_cuda/build-f32 --output-on-failure` passes 5/5.
- `python3 -m py_compile` passes for the Julia fixture validator and all Python
  scratch tools.
- `code/peps_cuda/tools/check_cuda_env.sh` runs locally and reports the expected
  missing `module`, `nvcc`, and `nvidia-smi` on this Mac. CUDA compilation and
  GPU smoke tests remain deferred to A100/JUPITER because this host has no CUDA
  toolchain or GPU.

## Implementation Checkpoint: 2026-05-15 01:30 CEST

Strengthened fixture/regression bridge:
- Extended the Julia fixture exporter with `native_axis_labels`,
  `theta_site_dims`, `theta_axis_labels`, and `sample_row_major`. This records
  the ITensor link identity and avoids confusing Julia's column-major
  `vec(sample)` with the C++ site-major spin layout.
- The regenerated fixture validator now independently reconstructs every small
  fixture row by enumerating link-label assignments. It checks `logpsi_exact`,
  the exported `O_k` prefix, and `||O_k||^2` for the five PEPS sample rows:
  real `D=1`, real `D=2` zero/checker, and complex `D=2` zero/checker.
- Added `research/peps_cuda/julia_fixture_axis_mapping.md` to document the
  Julia column-major theta convention, C++ tensor order, link-label direction
  inference, and why `sample_row_major` is needed.
- Embedded the Julia fixture cases in the C++ unit tests:
  `real_3x2_D1_zero_sample`, `real_3x2_D2_zero_sample`, and
  `complex_3x2_D2_zero_sample`.
- The C++ `D=2` fixture tests explicitly transpose Julia theta-order data into
  C++ `SiteTensor` order and transpose C++ gradients back to Julia `O_k` order.
  This caught the expected storage-order mismatch during implementation; the
  final tests now validate both amplitude contraction and gradient ordering.
- The complex `D=2` fixture additionally checks phase-sensitive `logpsi` and
  holomorphic `O_k` conventions.
- Added nonzero checker-pattern samples for the real and complex `D=2`
  fixtures. These exercise `sample_row_major` and catch Julia/C++ sample-order
  mistakes. The real checker row has a negative real amplitude, so it also
  checks the principal-log phase convention (`imag(logpsi)=pi`).
- Added `code/peps_cuda/tools/run_cpu_regression.sh`, which configures/tests the
  FP64 and FP32 CPU builds, compiles Python validators, validates Julia
  fixtures, and records the local CUDA environment state.

Latest reference-export memory:
- Julia fixture export after adding checker samples:
  `128.58 real`, `1494384640` maximum resident set size, `1516128128` peak
  memory footprint.
- Fixture validator is fast and local; it reconstructs the `D=2` rows by
  enumerating 7 virtual links for the `3x2` lattice.

Latest local regression:
- `code/peps_cuda/tools/run_cpu_regression.sh` passes.
- FP64 CTest: 5/5 passing.
- FP32 CTest: 5/5 passing.
- Fixture validator: `rows=6 logpsi_rows=5 reconstructed_rows=5
  boundary_errors=0 max_env_error=1.332e-15`.
- Local environment probe still reports no `nvcc`/`nvidia-smi`, as expected.

Additional CPU memory trace:
- `3x3,D=2,Ns=20` Heisenberg FP64:
  parameters `128`, sampled-sector parameters `64`, dense `O` `40960` bytes,
  sampled-sector `O` `20480` bytes, peak RSS about `1.55 MiB`.
- Same case in FP32:
  dense `O` `20480` bytes, sampled-sector `O` `10240` bytes, peak RSS about
  `1.55 MiB`. At this tiny scale process/runtime overhead dominates RSS, but
  explicit `O` storage halves exactly as expected.
- `3x3,D=2,Ns=20` TFI exercises single-site flip buckets:
  first-sample buckets `diagonal=12`, `single=9`.
- Added `research/peps_cuda/precision_decision_matrix.md` with the planned
  FP64/FP32/TF32/lower-precision comparison metrics, suggested initial
  acceptance bands, and the thesis argument that FP32 helps memory but does not
  remove the need for direct/streamed Gram accumulation.
- Rechecked the same memory argument against an H200-style 141 GB HBM target:
  `32x32,D=8,Ns=5000` compact sampled-sector `O` is still about `156.54 GiB`
  in complex FP32 before workspaces/buffers, so H200 does not remove the direct
  Gram requirement.
- Updated A100 and JUPITER/GH200 Slurm smoke scripts to run
  `tools/check_cuda_env.sh` before building/profiling, so first cluster logs
  capture module/compiler/GPU context automatically. `bash -n` passes for both
  Slurm templates and both shell helper scripts.

## 2026-05-15 01:50 CEST Checkpoint

Additional validation/handoff work after the context handoff:

- Re-opened the external anchors most likely to drift before thesis citation:
  arXiv main paper and recent PEPS-tVMC/Ozaki papers, JUPITER configuration and
  GPU docs, NVIDIA Hopper/CUDA best-practice docs, cuBLAS grouped GEMM, and
  Simon Boehm's CUDA matmul worklog. `source_inventory.md` now records the
  2026-05-15 recheck.
- Added `program_lifetime_trace.md`, mapping the Julia
  `get_sample -> get_logpsi_and_envs -> get_all_horizontal_envs -> get_Ek ->
  get_Ok` object lifetimes to the C++/CUDA profiler trace target.
- Added `benchmark_triage_snapshot.md`, summarizing the GH200 one-GPU
  dense/compact/direct-Gram transition points from the benchmark matrix.
- Added `direct_gram_accumulation.md`, spelling out the sampled-sector minSR
  identity and the direct-Gram production plan.
- Ran a stricter compiler pass with `-Wall -Wextra -Wpedantic`; build and CTest
  pass.
- Ran a CPU sanitizer pass with AddressSanitizer/UBSan; build and CTest pass.
- Expanded Julia-generated fixture rows:
  `real_2x3_D2_striped_sample`, `complex_2x3_D2_striped_sample`, and
  `real_2x2_D3_checker_sample`.
- Regenerated Julia fixtures:
  `125.74 real`, maximum resident set size `1489371136`, peak memory footprint
  `1521420160`.
- The independent Python validator now reports:
  `rows=9 logpsi_rows=8 reconstructed_rows=8 boundary_errors=0
  max_env_error=1.332e-15`.
- The new `2x3`/`2x2,D=3` rows validate `logpsi` and boundary reconstruction
  but record a Julia reference `E/O` failure:
  `BoundsError: attempt to access 1-element Vector{QuantumNaturalfPEPS.Environment} at index [0]`.
  This is useful regression data rather than a blocker.
- Re-ran `code/peps_cuda/tools/run_cpu_regression.sh`: FP64 CTest 5/5, FP32
  CTest 5/5, fixture validator 8 reconstructed rows, local probe still reports
  no `nvcc`/`nvidia-smi`.
- `clang-format --dry-run --Werror` passes for C++/CUDA sources; `bash -n`
  passes for the Slurm and shell helper scripts.

## 2026-05-15 02:01 CEST Timer Close

- Timer lower bound satisfied: work continued past the requested 02:00 Berlin
  target.
- Final quick checks after 02:00:
  - Fixture validator:
    `rows=9 logpsi_rows=8 reconstructed_rows=8 boundary_errors=0
    max_env_error=1.332e-15`.
  - `bash -n` passes for A100/JUPITER Slurm scripts and shell helpers.
  - `clang-format --dry-run --Werror` passes for C++/CUDA sources.

## 2026-05-15 Julia CPU Profiling

- Added `code/peps_cuda/julia_reference/profile_reference_cpu.jl`.
- Profiled reduced versions of both upstream examples:
  `examples/heisenberg_multithreaded.jl` and `examples/CSL.jl`, plus synthetic
  `3x2`/`3x3` fixtures. The full upstream values
  (`Ns=1000,maxiter=10/4000`) were not run on the MacBook; the harness uses
  configurable `Ns=8,maxiter=1` for stage-level CPU profiling.
- Wrote `research/peps_cuda/julia_cpu_profile_report.md`.
- Single-thread `Ns=8` headline:
  - Heisenberg `4x4,D=2`: `71.9 ms` elapsed, `Oks_and_Eks` about `98.7%` of
    integrator time, `92.4 MiB` allocated in integrator.
  - CSL `4x4,D=2`: `91.7 ms` elapsed, `Oks_and_Eks` about `97.7%` of
    integrator time, `112 MiB` allocated in integrator.
  - In detailed single-thread timers, sampling is the largest stage; CSL also
    has a large four-body energy path.
- Threaded `Ns=8` headline:
  - Heisenberg `4x4,D=2`: `28.8 ms` elapsed, about `2.5x` over single-thread.
  - CSL `4x4,D=2`: `56.1 ms` elapsed, about `1.6x` over single-thread.
- Found instrumentation limitation: `Oks_and_Eks_threaded` does not pass
  `timer` down into `Ok_and_Ek`, so threaded mode loses detailed stage timings.
- Multiprocess profiling was attempted but not treated as a validated result:
  worker environment activation for the unregistered local packages needs more
  care, and Julia's main-process `Profile` is not a useful unified worker
  profiler anyway.
