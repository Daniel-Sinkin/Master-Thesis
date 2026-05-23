# Multi-GPU Strategy

Start simple: one MPI rank per GPU, samples sharded across ranks. Do not split a
single PEPS contraction across GPUs until sample sharding stops scaling.

## First Decomposition

For one optimization step:

```text
rank r owns samples [r * local_Ns, (r+1) * local_Ns)
all ranks hold PEPS tensors
all ranks either hold or receive the same double-layer environment version
rank r computes local samples, E_r, O_r / compact O_r
all ranks contribute to global minSR solve
all ranks or rank 0 apply theta update
```

This matches the algorithm because samples are independent once a usable
double-layer environment is available.

## What To Replicate

Replicate on every GPU at first:

- PEPS tensors.
- Hamiltonian term records.
- Current/stale double-layer environment version.
- Solver metadata and parameter layout.

This is acceptable because the PEPS tensor memory is small compared with dense
`O` and boundary/workspace buffers. Replication also avoids GPU-GPU traffic in
the contraction hot path.

## What To Shard

Shard:

- Samples.
- `E_loc`.
- Compact sampled-sector `O` rows or direct Gram tile work.
- Hamiltonian flip records expanded per sample.

Avoid sharding individual rows/sites at first. It creates fine-grained
communication and makes environment reuse harder.

## Gram Assembly Options

### Dense/Compact Row Allgather

Each rank computes local compact `O_r`, allgathers compact rows, then forms its
assigned tiles of `T = O O^dagger`.

Pros:

- Simple.
- Good first debug strategy for small `Ns`.

Cons:

- Compact `O` allgather can exceed HBM for production `D=8`, `32x32`.

### Distributed Gram Tiles

Each rank owns a subset of sample-pair tiles `(s,t)` and streams compact/direct
site slices needed for those tiles.

Pros:

- Avoids full compact `O` replication.
- Natural next step after single-GPU direct Gram.

Cons:

- More communication and scheduling complexity.

### Site-Sliced Allreduce

Each rank accumulates partial Gram contributions for the sites/samples it owns,
then allreduces the `Ns x Ns` Gram.

Pros:

- Gram is small: about 61 MiB for `Ns=2000`, 381 MiB for `Ns=5000` in complex
  FP64.
- Communication object is bounded by sample count, not parameter count.

Cons:

- Requires every rank to have enough information to compute its site slice
  contributions.

This is likely the production direction once direct site/environment Gram is
implemented.

## Applying `-O^dagger x`

For sample-sharded ranks:

```text
rank r computes local parameter update contribution:
  delta_theta_r = -O_r^dagger x_r
global delta_theta = allreduce_sum(delta_theta_r)
```

For compact sampled-sector rows, this is a scatter-add into the dense parameter
vector. The CUDA smoke kernel `launch_sampled_sector_minsr_apply_odag` is the
single-GPU version of this operation.

## Double-Layer Environment Refresh

Options:

1. Every rank refreshes the double-layer environments redundantly.
2. One rank refreshes and broadcasts to the node.
3. One GPU per node refreshes while other GPUs consume the previous version.

Start with redundant refresh for simplicity unless profiling shows it dominates.
The paper supports stale environments corrected by importance weights, so a
producer/consumer versioned environment queue is a reasonable next step.

## JUPITER Notes

JUPITER Booster nodes expose four GH200/H100-class GPUs. Slurm normally maps one
GPU to one task via `CUDA_VISIBLE_DEVICES`, so the first implementation should
not override that mapping.

Use:

- one rank per GPU,
- `--gpus-per-task=1`,
- `--cpus-per-task=72` on a full Booster node if using four ranks,
- `SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK`.

Keep hot tensors in HBM. Grace memory is useful for orchestration/staging, not
for hot contraction operands.

## Scaling Stop Points

Sample sharding is still the right strategy while:

- Per-rank sample count is large enough to keep the GPU busy.
- Gram/allreduce time is below contraction time.
- Double-layer refresh does not serialize all ranks.
- Compact/direct `O` memory fits per rank.

Consider finer-grained contraction sharding only if:

- `Ns` is too small per GPU,
- a single boundary-MPS contraction becomes too large for one GPU,
- or the direct Gram becomes dominated by repeated reads that cannot be fixed
  with tiling/reuse on one GPU.
