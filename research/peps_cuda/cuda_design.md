# CUDA Design Notes For Finite PEPS Sampling

This document is the bridge between the paper/Julia algorithm and a production
CUDA implementation. It is intentionally more concrete than the thesis prose and
less polished than a final chapter.

## Algorithm To Preserve

Reference Julia stage:

```text
generate_Oks_and_Eks
  for sample k:
    S, logpc, env_top = get_sample(peps)
    logpsi, env_top, env_down = get_logpsi_and_envs(peps, S, env_top)
    h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, S)
    Ek_terms = precompute Hamiltonian flip terms
    E_k = get_Ek(peps, terms, env_top, env_down, S, logpsi, h_envs_*)
    O_k = get_Ok(peps, env_top, env_down, S, logpsi, h_envs_*)
  solve minSR:
    theta_dot = -O^dagger (O O^dagger + lambda I)^-1 E
```

The C++/CUDA implementation should preserve that structure, but change the data
model:

- No ITensor dynamic index objects in the hot path.
- No per-contraction allocation in the sample loop.
- No one-launch-per-site contraction pattern.
- Explicit shape buckets and explicit tensor memory layout.

## Memory Layout

Use site-major, physical-slice-contiguous layout:

```text
site_tensor[site][physical][north][east][south][west]
```

Flattening:

```text
((((physical * north_dim + north) * east_dim + east) * south_dim + south)
 * west_dim + west)
```

Reasons:

- Sampling repeatedly projects physical slices, so each `T[site][s,:,:,:,:]`
  should be contiguous.
- `O` rows are sparse in the physical index: only the sampled physical slice
  contributes at each site.
- Horizontal row contractions sweep left-to-right and reuse east/west bonds.

Boundary sites keep real dimensions of 1 rather than padding all virtual legs to
`D`. Production GPU kernels can still use padded strides for alignment, but the
metadata should preserve true dimensions to avoid wasted edge work.

## Work Units

The minimum useful GPU task is not "one tensor contraction"; it is a bucket of
contractions with the same shape family:

- `double_layer_row_absorb`: build stale/current direct-sampling boundaries.
- `sample_row_density`: compute conditional reduced density matrices for a row.
- `single_layer_vertical_env`: top/down boundary-MPS for sampled single layer.
- `horizontal_env`: left/right contractions for each row.
- `ek_diagonal`: diagonal Hamiltonian terms, no flipped amplitude.
- `ek_horizontal_nearest`: one row, one or two neighboring sites.
- `ek_vertical_or_plaquette`: two adjacent rows and one/two columns.
- `ek_fallback`: rare exact/slow path.
- `ok_site`: tensor-removed gradient environment for sampled physical slices.
- `minsR_gram`: form `T = O O^dagger`.

## CPU Fallback Semantics

The exact C++ CPU path in `code/peps_cuda` is deliberately tiny-system:

- It contracts `Psi(S)` exactly by dynamic programming over row boundary states.
- It samples exactly by enumerating all computational-basis configurations.
- It computes `O` exactly by replacing one tensor with basis tensors.

This is not a boundary-MPS replacement. It is the semantic oracle for:

- Hamiltonian term expansion and signs.
- Complex phase behavior.
- `E_loc = <S|H|Psi>/Psi(S)`.
- `O_{S,i} = d Psi(S)/d theta_i / Psi(S)`.
- The minSR algebra.

## Boundary-MPS Production Direction

The production contraction layer should use a row-MPS representation:

```text
BoundaryMPS row:
  tensors[j] has left bond chiL, right bond chiR, vertical-open dimension v
```

Absorbing a PEPS row is an MPO application:

```text
boundary[j](aL, aR, n) * T[j](p fixed, n, e, s, w)
  -> enlarged boundary with horizontal bond chi * D
```

Then compress `chi * D -> Dc`.

First implementation:

- Use cuBLASLt/grouped GEMM for the local contractions.
- Use cuSOLVER or a CPU fallback for early SVD/eigendecomposition if GPU SVD
  setup takes too long.
- Keep compression modular: density-matrix algorithm first, SVD second, no
  custom eigensolver until profiling proves it matters.

Later implementation:

- For fixed small `D <= 8`, write custom CuTe/CUTLASS kernels for the most
  repeated contraction shapes.
- On Hopper, investigate TMA only after the GEMM-backed version is profiled.

## Direct Sampling Dataflow

The paper's Appendix B direct sampler can be GPU-shaped like this:

```text
for optimization step:
  maybe refresh double-layer boundaries on stream env_stream
  for sample batches:
    wait for a usable double-layer version
    for row i:
      build conditional row ket using previous projected top boundary
      build right-to-left unsampled row environment
      for column j:
        compute rho_j diagonal
        sample physical spin
        update left sigma
      update sampled top single-layer boundary
```

The awkward part is RNG and sequential dependence inside a sample. The parallel
axis is therefore samples, not sites, until row-level kernels are large enough.
For `Ns=1000..5000`, one GPU can host many sample workers.

## E And O Splitting

`E`:

- Precompute Hamiltonian terms on CPU.
- For each sample, expand terms into flip contributions.
- Classify by changed-site geometry, not by original operator geometry.
- Diagonal terms are just scalar accumulation.
- Horizontal terms reuse `h_envs_l/r` for that row.
- Plaquette/vertical terms reuse two-row environments.

`O`:

- `O` is logically dense but structurally sparse per site: for each site only
  the sampled physical sector is nonzero.
- Do not materialize all zero physical sectors if the minSR Gram can consume a
  sparse/sliced representation.
- First GPU version can materialize sample-major dense `O` because it simplifies
  `T = O O^dagger`; this may be memory-heavy:
  `Ns * Lx * Ly * d * D^4 * sizeof(complex)`.
- Better version accumulates `T` directly from per-site nonzero slices:
  `T_ss' += sum_site dot(O_site[s], O_site[s'])`.
- Current scaffold has the intermediate representation implemented: compact
  sampled-sector rows store one physical slice per site, `T_ss'` skips sites
  where the two samples selected different physical values, and the solved
  sample-space vector is scattered back into the dense parameter update. This is
  closely aligned with the "small-o trick" described in the recent Wu/Nys
  PEPS-tVMC work.

## Precision Policy

Start with complex FP64 for correctness.

Then test:

- FP64 accumulation with FP32 tensor storage for exploratory optimization.
- TF32/BF16 only for contractions whose error is damped by sampling noise.
- Always report variational/energy drift against FP64 CPU or high-accuracy GPU
  reference on tiny lattices.

The Blackwell FP64-emulation papers are useful background, but this project
targets A100 and GH200/H100/H200, so the first precision experiments should use
native Hopper/Ampere capabilities.

## Profiling Order

1. Nsight Systems:
   - Are there CPU gaps?
   - Is launch count absurd?
   - Are double-layer refreshes overlapping sample generation?
   - Are host-device copies present inside iteration?
2. Nsight Compute on one kernel family at a time:
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   - `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`
   - `sm__warps_active.avg.pct_of_peak_sustained_active`
   - `smsp__warps_eligible.avg.per_cycle_active`
   - `smsp__issue_active.avg.pct_of_peak_sustained_active`
   - branch/warp execution metrics for `E` kernels
   - tensor-core pipe metrics for GEMM-backed contractions

## First Cluster Experiments

Minimal sequence once a CUDA machine is available:

1. Build with `-DCMAKE_CUDA_ARCHITECTURES=80` on A100.
2. Build with `-DCMAKE_CUDA_ARCHITECTURES=90` on GH200/H100/H200.
3. Run CPU exact `2x2`, CUDA naive `2x2`, compare `Psi`, `E`, `O`.
4. Run synthetic contraction buckets with fixed `D,Dc,Ly` and random tensors.
5. Compare naive looped cuBLAS, strided batched GEMM, grouped GEMM, and custom
   kernel for the same bucket.
6. Only then plug into sampling.
