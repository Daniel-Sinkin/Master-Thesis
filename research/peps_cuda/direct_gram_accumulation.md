# Direct Sampled-Sector Gram Accumulation

This is the production replacement for materializing `Oks`. Dense and compact
`O` are still valuable debug objects, but thesis-scale runs need the Gram and
right-hand-side directly.

## Algebra

For samples `s,t`, minSR needs:

```text
T[s,t] = sum_i conj(O[s,i]) * O[t,i] + lambda * delta[s,t]
b[s]   = E[s]
theta_dot = - O^dagger * solve(T, b)
```

The Julia `get_Ok` path writes zero for every physical sector that was not
sampled. Therefore, with a site-wise layout:

```text
T[s,t] = sum_site indicator(spin[s,site] == spin[t,site])
         * dot(conj(Ocompact[s,site]), Ocompact[t,site])
```

This identity is already implemented in the C++ CPU tests and the CUDA smoke
kernel for compact sampled-sector rows.

## Why Direct Beats Compact Storage

Compact sampled-sector storage halves dense `O` for spin-1/2 PEPS, but the
memory still scales as:

```text
Ns * sum_site site_virtual_parameters * sizeof(complex)
```

At `32x32,D=8,Ns=5000`, this is still hundreds of GiB in FP64 and above one
JUPITER GH200 GPU even in FP32. Direct accumulation changes the dominant live
object to:

```text
sample-space Gram: Ns^2
sample vector:     Ns
parameter update:  Np, streamed/tiled
current O slice:   batch_samples * site_slice
```

For `Ns=5000`, the complex FP32 Gram is about `190.7 MiB` and the complex FP64
Gram is about `381.5 MiB`, which is small compared with `O`.

## First GPU Tiling

Start with a conservative tile:

```text
sample tile S_TILE x S_TILE
site loop outside or inside kernel family
within-site vector loaded in chunks
```

Candidate mapping:

- one thread block computes a small tile of `(s,t)` Gram entries,
- loop over sites,
- skip a site for `(s,t)` if sampled physical sectors differ,
- reduce virtual-slice dot products in shared memory or warp reductions,
- write/update Hermitian Gram entries.

For debugging, keep a separate kernel per site bucket. For performance, consider
accumulating by site into a tile buffer and reducing over sites, because the
branch pattern is site-local.

## Weighted minSR

For importance weights, use:

```text
Ow[s,*] = sqrt(w[s]) * O[s,*]
Ew[s]   = sqrt(w[s]) * E[s]
```

Then the same direct Gram applies:

```text
T = Ow * Ow^dagger + lambda I
```

In the apply path, either pre-scale the solved sample vector by `sqrt(w[s])` or
fold the scaling into the site-slice scatter. The current CUDA header documents
the pre-scale convention for the sampled-sector apply kernel.

## Numerical Checks

Every new direct-Gram implementation should be compared against:

1. dense `O` Gram on tiny random rows,
2. compact sampled-sector Gram on tiny random rows,
3. Julia fixture rows after tensor-order transposition,
4. Hermitian symmetry and non-negative diagonal after ridge shift,
5. minSR direction cosine against dense CPU FP64.

## Expected Bottlenecks

- Reading site slices repeatedly for all sample-pair tiles.
- Divergence from sample-pair spin-sector checks.
- Atomic or reduction overhead if many blocks update the same Gram tile.
- Reconstructing `Ocompact` from environments too slowly.

If sample-pair divergence is severe, group samples by spin bitmasks per site or
accumulate one site at a time with contiguous sample lists for each physical
sector. That turns the condition into two dense blocks for spin-1/2.

## Preferred First Production Path

1. Build compact `Ocompact` for a batch and compare direct Gram to materialized
   compact Gram.
2. Replace persistent compact `Ocompact` with site-streamed slices.
3. Keep only the current site/batch slice plus the Gram tile live.
4. Add multi-GPU sample sharding: each rank accumulates local Gram blocks or
   local `Odagger_x`, then allreduces the small objects.

The important point: this optimization is algorithmic memory reduction first
and kernel tuning second.
