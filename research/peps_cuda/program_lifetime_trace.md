# Program Lifetime Trace

This is the runtime object/lifetime view of the Julia reference path in
`QuantumNaturalfPEPS.jl-main/src/Ok_and_Ek.jl`, `sampling.jl`, `Ok.jl`, and
`Ek.jl`. It is written as the map to compare against the C++/CUDA profiler
trace.

## One `Ok_and_Ek` Sample

The reference sequence is:

```text
get_sample
  -> get_logpsi_and_envs
  -> get_all_horizontal_envs
  -> precompute Hamiltonian flip terms
  -> get_Ek
  -> get_Ok
```

Important lifetime consequence: the sample, vertical environments, horizontal
environments, flipped-amplitude cache, local energy, and full gradient row are
all simultaneously live near the end of the sample. In a GPU implementation,
this is the point where materializing dense `O` becomes the dominant memory
mistake.

## Sampling Stage

`get_sample(peps; mode=:full)` allocates:

- `S`, the integer spin sample matrix,
- `env_top`, one top boundary environment per row except the last,
- per-row `ket`,
- per-row `bra`,
- per-row unsampled right environments `E`,
- per-site reduced density matrix `rho_r`,
- evolving `sigma`, the contraction of already sampled sites in the row.

For `mode=:fast`, sampled top environments are updated from the sampled ket. For
`mode=:full`, top environments are recomputed row by row from projected PEPS
rows. The GPU path should keep both modes explicit:

- `full`: correctness and reference-quality samples,
- `fast/stale`: throughput mode with importance correction.

Memory/profiling markers to add later:

```text
sample/ket_row
sample/unsampled_row_env
sample/rho_site
sample/update_sigma
sample/top_env_full
sample/top_env_fast
```

## Vertical Environments

`get_logpsi_and_envs` recomputes projected single-layer environments for the
drawn sample, returning:

- `logpsi`,
- `env_top`,
- `env_down`,
- `max_bond`.

This is the reference point for contraction accuracy. The C++ CPU exact oracle
does not approximate this; it enumerates virtual labels, so tiny fixture
agreement proves index/sign conventions but not boundary-MPS truncation
quality.

## Horizontal Environments

`get_all_horizontal_envs` creates left/right row environments for the fixed
sample and vertical boundaries. These are reused by both:

- `get_Ek`, for flipped-amplitude ratios,
- `get_Ok`, for per-site log-gradient tensors.

GPU rule: horizontal environments are short-lived but extremely valuable. They
should feed `E` and compact `O` immediately, then be released or reused across
samples in a bucket. Keeping all horizontal environments for a large sample
batch is likely worse than recomputing some cheap pieces.

## Energy Stage

The Julia path builds `Ek_terms` via
`QuantumNaturalGradient.get_precomp_sOpsi_elems(...; get_flip_sites=true)`, then
sorts flip terms into approximate geometry classes:

- diagonal/no flip,
- horizontal/single-site,
- vertical,
- four-body plaquette,
- longer horizontal,
- fallback/other.

Observed reference bug: `sort_dict` inserts vertical terms into `vetr`, which is
undefined, instead of `vert`. Preserve fixture behavior where possible, but fix
the C++ bucket classifier because its role is profiler layout rather than
literal dynamic-dispatch parity.

GPU rule: flatten each Hamiltonian contribution to a typed record before the
sample loop. Dynamic dictionaries of tuples are a CPU convenience and a GPU
launch anti-pattern.

## Gradient Stage

`get_Ok` loops over every active PEPS site:

1. contract all environments except the site tensor,
2. scale by `exp(f - logpsi)`,
3. write that tensor into only the sampled physical sector,
4. fill every unsampled physical sector with zeros,
5. advance the dense global parameter offset.

This is the main structural gift in the reference implementation: even though
it writes a dense vector, the nonzero pattern is sampled-sector sparse. The
baseline C++ code implements both dense rows and compact sampled-sector rows;
the production GPU path should use compact/direct accumulation.

## Peak Memory Expectation

For one process, peak memory is expected after `get_Ok` has produced `Oks` for
many samples and before minSR releases intermediate matrices:

```text
PEPS parameters
+ double-layer environments
+ current sample vertical/horizontal environments
+ sample matrix/logpsi/logpc/E arrays
+ Oks matrix
+ sample-space Gram
+ solver workspace
```

For a single sample, environments can dominate. For many samples, `Oks`
dominates unless direct Gram accumulation is used. That is why the C++ demo
prints both dense-`O` and sampled-sector-`O` byte counts and why cluster runs
should add NVTX ranges around each allocation stage.

## C++/CUDA Trace Target

The intended production trace should instead look like:

```text
initialize resident PEPS/Hamiltonian
refresh double-layer boundaries
for sample-batch:
  sample rows/sites
  compute single-layer vertical/horizontal environments
  accumulate E
  accumulate compact/direct Gram tiles
  accumulate Odagger_x tiles
solve sample-space minSR
apply/scatter parameter update
```

The key absence in a good trace is a large persistent dense `Oks` allocation.
