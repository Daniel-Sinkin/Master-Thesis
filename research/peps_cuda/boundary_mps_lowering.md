# Boundary-MPS CUDA Lowering Notes

This note records the concrete index transformations for the row-boundary
contractions. The symbols are implementation symbols, not final thesis notation.

## Single-Layer Row Absorption

Assume the PEPS tensor for one sampled site is already projected to a physical
slice:

```text
A_j[n, e, s, w]
```

where `n/e/s/w` are the north/east/south/west virtual legs. The current boundary
MPS tensor at column `j` is

```text
B_j[aL, aR, n]
```

where `aL/aR` are boundary-MPS horizontal bonds and `n` is the open vertical leg
that contracts with the PEPS row. Absorbing one PEPS row gives an enlarged
boundary tensor

```text
C_j[(aL,w), (aR,e), s] =
    sum_n B_j[aL, aR, n] * A_j[n, e, s, w]
```

The horizontal bonds therefore expand from `chi` to roughly `chi * D` before
compression.

### GEMM View

For a fixed column and fixed true edge dimensions:

```text
B_mat[(aL,aR), n]      shape: (chiL * chiR) x nDim
A_mat[n, (w,e,s)]      shape: nDim x (wDim * eDim * sDim)
C_mat[(aL,aR), (w,e,s)] shape: (chiL * chiR) x (wDim * eDim * sDim)
```

Then reshape/permute:

```text
C_mat[(aL,aR), (w,e,s)]
  -> C[(aL,w), (aR,e), s]
```

This is a good candidate for grouped GEMM because edge columns have different
`w/e/n/s` dimensions, and compression changes `chiL/chiR` along the sweep.

## Double-Layer Row Absorption

For sampling environments, use the double-layer local tensor:

```text
M_j[(n,n'), (e,e'), (s,s'), (w,w')] =
    sum_p conj(A_j[p,n',e',s',w']) * A_j[p,n,e,s,w]
```

or a partially projected version when some spins in the row have already been
sampled. The boundary tensor is then

```text
B2_j[aL, aR, (n,n')]
```

and row absorption is the same pattern:

```text
C2_j[(aL,(w,w')), (aR,(e,e')), (s,s')] =
    sum_(n,n') B2_j[aL,aR,(n,n')] *
               M_j[(n,n'),(e,e'),(s,s'),(w,w')]
```

The important difference is that virtual dimensions are effectively squared.
This explains the much larger double-layer cost and why stale/reused
double-layer environments are algorithmically important.

## Compression

After absorption, an MPS horizontal bond has grown by `D` in the single layer or
by `D^2` in the double layer. Compression should be a separate module:

```text
compress_boundary(B, target_chi=Dc)
```

First candidate algorithms:

- QR/SVD sweep on GPU with cuSOLVER for larger matrices.
- Density-matrix compression if that maps better to Hermitian eigensolvers.
- CPU compression only for tiny debug cases.

The compression API should not care whether local absorption was implemented by
looped GEMM, grouped GEMM, cuTENSOR, or a custom kernel.

## Horizontal Environments For `E` And `O`

Once top/down single-layer boundaries are available for a sample, each row also
needs left/right horizontal environments. For a fixed row:

```text
L_j = contraction of columns < j
R_j = contraction of columns > j
```

The site-gradient environment is then roughly:

```text
G_j[p,n,e,s,w] =
    top/down/left/right environment with site tensor A_j[p,n,e,s,w] removed
```

The log-gradient row stores

```text
O_j[p,n,e,s,w] = G_j[p,n,e,s,w] / Psi(S)
```

but for a sample `S`, only `p = S_j` is nonzero. The compact sampled-sector
layout therefore stores:

```text
Ocompact_j[n,e,s,w] = G_j[S_j,n,e,s,w] / Psi(S)
```

The dense parameter update can be recovered later by scattering compact slices
back into the physical sector selected by each sample.

## Hamiltonian `E` Buckets

For `E_loc(S)`, each off-diagonal Hamiltonian contribution requires a ratio

```text
Psi(S_flipped) / Psi(S)
```

The expensive part is recomputing the local region affected by the flip. Bucket
by changed support:

- diagonal: no contraction, scalar only;
- single-site: reuse all row environments around one site;
- horizontal nearest: reuse one row's left/right environments;
- vertical nearest: reuse two adjacent row boundaries;
- plaquette: reuse two rows and two columns;
- horizontal long: special same-row handling;
- fallback: generic slower path.

The Julia `sort_dict` idea should become an explicit array-of-records layout per
bucket so GPU kernels do not branch through general operator structures.

## First Shape Buckets To Benchmark

Use synthetic tensors before wiring the full sampler:

```text
single_absorb:
  M = chiL * chiR
  K = nDim
  N = wDim * eDim * sDim

double_absorb:
  M = chiL * chiR
  K = nDim^2
  N = wDim^2 * eDim^2 * sDim^2

gradient_site:
  output = pDim * nDim * eDim * sDim * wDim

sampled_sector_gram:
  for each pair (s,t), sum only sites where S_s[j] == S_t[j]
```

For each family, benchmark:

- looped cuBLAS,
- strided batched GEMM when shapes are identical,
- grouped GEMM when edge/interior shapes differ,
- cuTENSOR when avoiding transposes is valuable,
- custom CuTe/CUTLASS kernels only after the library baselines are measured.

## Data-Layout Rules

- PEPS tensors: site-major, physical-major, then `n/e/s/w`.
- Projected sampled slices: sample-major, site-major, compact virtual slice.
- Boundary tensors: keep the leg to be contracted contiguous where possible.
- Shape metadata: store true edge dimensions and padded strides separately.
- No allocation inside sample, `E`, `O`, or boundary absorption loops.
- Prefer one launch per shape bucket over one launch per site.
