# Reference Algorithm Anchors

These are the implementation facts I want close at hand while translating the
paper and Julia reference into CUDA/C++.

## Paper Anchors

### minSR

The paper moves stochastic reconfiguration from parameter space to sample space:

```text
parameter-space: G = O^dagger O, shape Np x Np
sample-space:    T = O O^dagger, shape Ns x Ns
```

For finite PEPS, the parameter count is:

```text
Np = Lx * Ly * D^4 * d
```

So for realistic `D=8`, `Np` is huge compared with `Ns=1000..5000`. This is why
the implementation should optimize generation and consumption of sample rows,
not build `G`.

The paper's derivation uses the raw ratio rows
`O_{S,i} = (partial Psi(S) / partial theta_i) / Psi(S)` and
`E_loc(S) = <S|H|Psi>/Psi(S)` in the sampled least-squares problem. It notes
that explicit wavefunction normalization would introduce an additional term.
The current CPU oracle follows this raw convention; centered/gauge-fixed variants
such as in other tVMC work should be treated as optional solver variants, not
silently mixed into this reference path.

## Local PEPS Structure PDF Anchors

The local `PEPS_structure_documentation.pdf` restates the same least-squares
problem and explicitly lists the tasks:

```text
evaluate Psi(S)
evaluate dPsi(S)/dtheta
solve for theta_dot
generate samples S and p(S)
```

It also gives the practical regimes:

- Imaginary time: `8x8` to `32x32`, `D=2..8`, usually `Ns ~ 1000..5000`.
- Real time: smaller lattices around `8x8`, but much larger sample counts, often
  `Ns ~ Np`.

Small caveat: in Sec. 3.3 the PDF says that when `Ns << Np` one typically uses
Eq. 12, but Eq. 12 in that document is the parameter-space inverse. The
surrounding explanation says the linear system is small and does not grow with
system size, so this appears to mean the sample-space/minSR equation Eq. 13.

### Boundary-MPS Costs

The paper gives the relevant scaling estimates:

```text
single-layer: O(Dc^3 D^3) + O(Dc^2 D^4)
double-layer: O(Dc^3 D^4) + O(d Dc^2 D^6)
```

CUDA implication:

- Single-layer contractions dominate repeated `Psi(S)`, `E`, and `O` work.
- Double-layer contractions are heavier per refresh but can use smaller
  `Dc_double ~ D` for direct sampling and can be reused/stale.

### Direct Sampling

Appendix B uses conditional row density matrices:

```text
for row i:
  build T^u_{S_<i}[i] from previous sampled rows
  combine with lower double-layer boundary D^l[i+1]
  sample row S_i from conditional probabilities
  update sampled upper boundary E^u[i]
```

Implementation consequences:

- Sampling is sequential inside one sample.
- Samples are independent once a usable double-layer environment version exists.
- The parallel GPU axis is therefore sample batches first.
- The sampler must return `logpc` so importance weights can correct the
  approximate sampling probability.

### Stale Double-Layer Environments

The paper explicitly supports asynchronous/stale double-layer boundaries with
importance correction. It reports a `16x16`, `D=8`, `Ns=2000` case where the
double-layer environments can lag by about five optimization steps without
significant error-metric changes.

CUDA implication:

- Treat double-layer refresh as a separate stream/work queue.
- Version the environment used by each sample batch.
- Track weights/energy variance to detect stale-environment failure.

## Wu/Nys Small-o Anchor

The Wu/Nys PEPS-tVMC paper describes the same physical-sector sparsity in a
useful notation:

```text
O[x](s)[p,l,r,d,u] = 0 if p != s(x)
o[x](s)[l,r,d,u]  = O[x](s)[s(x),l,r,d,u]
```

The minSR matrix can be reconstructed from `o` by masking sample pairs according
to whether their physical value at the corresponding site/parameter block
matches. This is exactly what `sampled_sector_gram` and
`launch_sampled_sector_minsr_gram` do in the current scaffold.

### Diagonal Long-Range Terms

Long-range diagonal interactions are cheap in the sampling estimator because
they do not require flipped amplitudes. This matters for Rydberg-style models:
the `E` path should keep diagonal one-site and two-site kernels separate from
off-diagonal contraction buckets.

## Julia Anchors

### `Ok_and_Ek.jl`

Reference sequence:

```text
get_sample
get_logpsi_and_envs
get_all_horizontal_envs
get_Ek
get_Ok
```

The C++/CUDA scaffold preserves this stage order, but replaces dynamic ITensor
objects with packed arrays, explicit metadata, and bucketed kernels.

### `Ok.jl`

The Julia gradient writes zeros for all unsampled physical sectors and writes
the site environment into the sampled sector. This directly motivates the
compact sampled-sector representation now implemented in the scaffold.

### `Ek.jl`

The Julia code sorts local-energy work into geometry buckets before evaluating
flipped amplitudes. CUDA should make this explicit:

```text
diagonal
single-site
horizontal-nearest
vertical-nearest
plaquette
horizontal-long
other/fallback
```

### `Distributed/Oks_and_Eks.jl`

The importance weights are computed from:

```text
log_ratio = 2 * real(logpsi) - logpc
logZ = logsumexp(log_ratio) - log(Ns)
weight = exp(log_ratio - logZ)
```

The C++ and CUDA weight helpers now use this normalization convention.

### `sampling.jl`

The direct sampler builds/reuses double-layer environments, constructs
right-to-left unsampled row environments, samples physical values from local
conditional density matrices, and updates the sampled upper boundary.

CUDA implication:

- Keep double-layer environment generation separate from per-sample work.
- Within a row, right-to-left unsampled environments are reusable across the
  left-to-right conditional sampling pass.
- Row-level sampling should be expressed as a structured kernel family or
  persistent sample worker, not as one tiny launch per site.

### `double_layer_async.jl`

The Julia async implementation keeps a shared copy of double-layer environments
and refreshes them in a background thread. CUDA should translate the idea, not
the mechanism:

```text
env_stream refreshes version k+1
sample streams consume latest complete version k
sample records store version/logpc
importance weights correct the mismatch
```
