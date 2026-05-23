# Memory Hierarchy Notes For PEPS CUDA

## The Practical Hierarchy

For the first production implementation, think in this order:

1. HBM:
   - PEPS tensors, projected slices, boundary-MPS environments, sample buffers,
     `E`, `O` slices, and Gram buffers must live here during the hot loop.
   - JUPITER GH200 lists about 4 TB/s HBM bandwidth per GPU; H200 SXM lists
     4.8 TB/s; A100 40GB is about 1.555 TB/s.
2. L2:
   - Hopper H100-class GPUs have 50 MB L2.
   - Reusing the same PEPS tensor slices across thousands of samples can benefit
     from L2 if sample workers are batched by site/row and access patterns are
     stable.
   - L2 persistence controls are worth testing only after kernel families are
     stable.
3. L1/shared memory:
   - Hopper exposes up to 228 KB shared memory per SM and 227 KB per block with
     opt-in.
   - Shared memory is valuable for repeated small contractions only when it
     raises arithmetic intensity enough to compensate for lower occupancy.
   - A100 has less shared memory per SM, so Hopper-friendly tile sizes may be too
     heavy for the A100 development path.
4. Registers:
   - Register tiling matters for contraction kernels.
   - Too many registers/thread cuts occupancy quickly because H100 and A100 both
     have 64K 32-bit registers per SM.
5. Grace LPDDR / host memory:
   - GH200 NVLink-C2C makes CPU-GPU access much better than PCIe.
   - Still do not use host/Grace memory for hot PEPS contraction operands.

## PEPS-Specific Data Placement

Keep in HBM:
- Packed PEPS tensors.
- Double-layer boundary-MPS environments, with versioning for stale async use.
- Single-layer top/down environments for current sample batch.
- Horizontal row environments needed by `E` and `O`.
- Sample-major spin buffers and log probabilities.
- Hamiltonian flip records grouped by bucket.
- Gram matrix and solver workspaces.

Keep on CPU/Grace:
- Hamiltonian construction and static term lists.
- Shape bucket planning.
- Logging/profiler metadata.
- Occasional validation samples.

Move rarely:
- PEPS parameter vector at optimizer step boundaries.
- Energy summaries and diagnostic reductions.
- Small solver outputs if a CPU solver is temporarily used for debugging.

## Access Patterns

PEPS tensor layout:

```text
site_tensor[site][physical][north][east][south][west]
```

Why:
- Sampling and `O` select one physical sector per site.
- Projection is a contiguous copy per sampled physical value.
- Future direct-Gram accumulation can address sampled physical sectors without
  walking zero sectors.

Sample layout:

```text
sample_spins[sample][site]
```

Why:
- Sample-parallel kernels read a contiguous site vector per sample.
- Site/bucket-parallel kernels can later transpose or tile if needed, but the
  first direct sampler is sample-major.

`O` layout:
- Baseline dense `O[sample][parameter]` for correctness.
- Production should use site-sliced sampled physical sectors:

```text
O_site[site][sample][within_sampled_physical_slice]
```

This makes direct Gram accumulation natural:

```text
if sample[s][site] == sample[s'][site]:
    T[s,s'] += dot(O_site[site][s], conj(O_site[site][s']))
```

The current scaffold stores compact sampled-sector rows sample-major for
simplicity and smoke tests. A later production direct-Gram kernel may transpose
to site-major blocks to improve reuse across sample-pair tiles.

## TMA And Hopper-Only Features

Use Hopper TMA when:
- The same multidimensional tile movement repeats many times.
- Shared-memory tiles are large enough that register-based copy overhead matters.
- A100 fallback is already working and profiling says memory movement is a
  bottleneck.

Do not start with TMA:
- Boundary-MPS contraction shapes and compression workflow are not frozen yet.
- cuBLASLt/grouped GEMM and cuTENSOR may already handle the bottleneck well.

## Occupancy Scratchpad Examples

The `occupancy_scratch.py` helper is only a ceiling estimator, but it is useful
for catching tile choices that are obviously too heavy before an Nsight run.

Example ceilings:

| Arch | Threads | Regs/thread | Smem | Active blocks/SM | Active warps/SM | Occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 SM80 | 256 | 64 | 48 KiB | 3 | 24/64 | 37.5% |
| H100/GH200 SM90 | 256 | 64 | 48 KiB | 4 | 32/64 | 50.0% |
| H100/GH200 SM90 | 128 | 96 | 96 KiB | 2 | 8/64 | 12.5% |

Interpretation: a Hopper-only tile can spend much more shared memory than an
A100 tile, but large shared-memory tiles quickly become latency-sensitive. For
the first A100 development path, prefer smaller tiles and library baselines; for
SM90, only move to larger shared/TMA tiles after the grouped-GEMM/cuTENSOR
baseline identifies a fixed hot shape.

## Common Failure Modes

- Materializing dense `O` for large lattices:
  `32x32, D=8, d=2, Ns=5000` is about 625 GiB in complex FP64.
- Assuming sampled-sector `O` is enough:
  for `d=2`, sampled sectors still require about 312.5 GiB for that same
  `32x32, Ns=5000` case.
- One kernel launch per tiny contraction:
  this will show up as Nsight Systems CPU gaps and low GPU occupancy.
- Per-sample host/device copies:
  this destroys any benefit from HBM bandwidth.
- Treating unified memory as automatic optimization:
  GH200 makes migration less catastrophic, not free.
- Overusing shared memory:
  lower occupancy is fine only if arithmetic intensity rises enough.
