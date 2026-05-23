# PEPS CUDA Performance Plan

This is the practical profiling plan for the first A100/H100/GH200 runs. It is
written as a working checklist, not thesis prose.

## Baseline Assumptions

- Initial correctness precision: complex FP64.
- Initial production target: one MPI rank per GPU, sample batches sharded across
  GPUs.
- Primary target machine: JUPITER Booster GH200 nodes, 4 GPUs per node.
- Development target: A100 first, then H100/GH200. H200 is useful as a separate
  cloud target, but current JUPITER docs indicate 96 GB HBM3 GH200/H100 GPUs,
  not H200 SXM.
- The learning `tensor-network` code in this repo is not a production baseline.

## Stage 0: Correctness Harness

Goal: prove C++/CUDA semantics match the Julia/paper pipeline on tiny lattices.

Required checks:
- `Psi(S)` exact CPU vs CUDA exact/small path.
- `E_loc(S)` for diagonal and off-diagonal local terms.
- `O_{S,i}` physical-sector sparsity and phase.
- `theta_dot = -O^dagger (O O^dagger + lambda I)^-1 E`.
- Parameter-space SR and sample-space minSR agree on tiny problems under the
  same ridge regularization.
- Sampling probabilities and importance weights.

Acceptance:
- `2x2` and `3x3` exact tests pass in FP64.
- Energy/log-gradient differences are at FP64-level tolerance before approximate
  boundary-MPS compression is introduced.

## Stage 1: Synthetic Shape Buckets

Goal: understand contraction performance before wiring the full sampler.

Buckets:
- Single-layer row absorption for fixed sampled physical slices.
- Double-layer row absorption for sampling environments.
- Horizontal left/right environment construction.
- `O` site-gradient environment contraction.
- Plaquette/vertical flipped-amplitude contraction.
- Dense minSR Gram formation.
- Sampled-physical-sector minSR Gram formation.

For each bucket benchmark:
- Naive loop over cuBLAS calls.
- Strided batched GEMM if all shapes match.
- Grouped GEMM if shapes vary.
- cuTENSOR for awkward dense tensor contractions.
- Hand-written CUDA only after library baselines are measured.

Record:
- Shape metadata: `M,N,K,batch`, data type, transposes, stride/padding.
- Effective TFLOP/s.
- Effective GB/s.
- Kernel launch count.
- Workspace allocation count and bytes.

## Stage 2: Boundary-MPS Implementation

Goal: replace exact CPU contraction with approximate row-MPS contraction.

Initial algorithm:
- Represent a boundary row as MPS tensors with explicit left/right/vertical
  dimensions.
- Absorb one projected PEPS row as an MPO application.
- Compress `chi * D -> Dc` after each row.
- Start with density-matrix/SVD compression, keeping the compression API
  replaceable.

GPU lowering:
- Local contractions become GEMM/grouped GEMM buckets.
- SVD/eigendecomposition can initially call cuSOLVER or run on CPU for tiny
  debugging, but production should keep it on GPU.
- Keep all boundary buffers pooled and reused. No allocation in the sample loop.

Acceptance:
- Increasing `Dc` converges to exact CPU values on tiny lattices.
- Boundary-MPS result matches Julia within compression tolerance on small PEPS.

## Stage 3: Direct Sampling

Goal: implement Appendix-B style direct sampling with stale/reused double-layer
environments.

Parallel axis:
- Samples are independent once a usable double-layer environment version exists.
- Sites within one sample are sequential because conditional probabilities
  depend on earlier sampled spins.
- Rows have partial parallelism in the construction of right-to-left row
  environments.

Streams:
- `env_stream`: refresh double-layer environments.
- `sample_stream[k]`: sample batches using latest accepted environment version.
- `energy_stream`: evaluate `E` once sample and single-layer envs are ready.
- `gradient_stream`: evaluate `O`, possibly overlapping with `E` buckets.

Acceptance:
- Sampling distribution agrees with exact enumeration on tiny systems.
- Importance weights remain numerically stable under stale environment reuse.

## Stage 4: E/O Splitting

`E` plan:
- Precompute Hamiltonian terms on CPU.
- Expand terms for each sample into changed-site flip contributions.
- Bucket by changed support: diagonal, single-site, horizontal-nearest,
  vertical-nearest, plaquette, horizontal-long, fallback.
- Diagonal terms never launch contraction kernels.
- Horizontal/plaquette terms reuse horizontal and vertical environments.

`O` plan:
- Baseline: materialize dense sample-major `O` for simple `O O^dagger`.
- Better: store only sampled physical sectors per site.
- Best: accumulate `T_ss' += sum_site dot(O_site[s], O_site[s'])` directly
  without materializing full zero sectors.

The current scaffold now has a sampled-sector Gram oracle/kernel: for each pair
of samples it only contracts a site's compact gradient slice when both samples
used the same physical value on that site. This saves the factor `d` relative to
dense `O` and validates the indexing convention, but it is still too large for
the hardest `D=8`, `32x32`, `Ns=5000` cases.

Dense `O` memory sanity:
- `Ns=2000`, `32x32`, `D=8`, `d=2` gives roughly
  `2000 * 32 * 32 * 2 * 8^4 * 16 bytes = 268 GB`.
- This does not fit a single JUPITER GPU and strongly argues for sliced or
  direct Gram accumulation.
- Storing only the sampled physical sector saves the factor `d`, but that is
  still about 134 GB for the same case and about 312.5 GiB for
  `Ns=5000, 32x32, D=8`. Direct Gram accumulation remains the likely endpoint.

## Stage 5: Profiler KPIs

Run Nsight Systems first:
- CPU gaps between kernels.
- Kernel launch count per sample and per optimization step.
- Host-device copies or unified-memory migrations inside the iteration.
- Stream overlap between double-layer refresh and sample/E/O work.
- MPI/NCCL/IB time once multi-GPU sharding starts.

Run Nsight Compute by bucket:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`
- `l1tex__t_sector_hit_rate.pct`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`
- Warp branch/execution efficiency for Hamiltonian bucket kernels.
- Tensor-core pipe activity for GEMM/cuTENSOR/cuBLASLt paths.

Interpretation:
- High launch count + CPU gaps: merge into grouped GEMM/CUDA graphs/persistent
  kernels.
- High DRAM and low SM utilization: improve reuse, padding, coalescing, or
  contraction order.
- High SM but low tensor-core use on GEMM-shaped kernels: revisit precision,
  cuBLASLt algorithms, and tensor-core eligibility.
- Low eligible warps with memory stalls: increase independent work per block,
  register tiling, or batching.

## Stage 6: Hardware-Specific Passes

A100:
- Treat as the portability baseline.
- Avoid Hopper-only features until the A100 path is stable.
- Use this to debug correctness and basic roofline behavior.

H100/GH200:
- Compile SM90 (`-DCMAKE_CUDA_ARCHITECTURES=90`).
- Test larger shared-memory carveouts for custom kernels.
- Investigate TMA only after stable hot kernels exist.
- Consider CUDA Graphs for repeated sample-loop structure once kernel topology
  stabilizes.
- Consider cuBLASDx/CuTe only for repeated tiny contractions that remain
  bottlenecks after cuBLASLt/grouped GEMM/cuTENSOR baselines.

JUPITER GH200:
- Keep hot tensors in HBM.
- Use Grace memory for orchestration and staging, not hot contraction buffers.
- Start with one rank/GPU and sample sharding; only split individual PEPS
  contractions across GPUs if sample sharding stops scaling.
- Be careful with Nsight Compute clock-locking behavior in JSC docs when
  comparing performance numbers.
