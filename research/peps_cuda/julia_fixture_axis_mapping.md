# Julia Fixture Axis Mapping

The reference fixture exporter stores enough metadata to reconstruct small
ITensor PEPS tensors outside Julia without guessing virtual-leg directions.

## Fields

- `theta`: flattened `vec(peps)` values from Julia.
- `theta_site_dims`: per-site dimensions in the exact axis order used by
  `vec(peps)`, namely `(siteind(peps,i,j), linkinds(peps,i,j)...)`.
- `theta_axis_labels`: string labels for those axes. Link labels contain the
  source and target coordinate, for example `1;1->1;2,h_link`.
- `native_axis_labels`: ITensor's native tensor axis order, retained only for
  debugging.
- `sample`: Julia `vec(sample)` order. This is column-major for a Julia matrix.
- `sample_row_major`: C++ site-major order, matching
  `site_index = x * ly + y`.

The checker fixtures deliberately use `sample_row_major = [0,1,1,0,0,1]`.
Their Julia `sample` field is `[0,1,0,1,0,1]`, demonstrating why both fields
are stored.

## Flattening Convention

Julia arrays are column-major. For a tensor with dimensions
`(d0,d1,d2,...)`, the offset is:

```text
offset = i0 + d0 * (i1 + d1 * (i2 + ...))
```

The current C++ `SiteTensor` storage is physical-major with the last virtual
index fastest:

```text
offset = ((((p * north + n) * east + e) * south + s) * west + w)
```

Therefore a `D>1` fixture import must both identify the direction of each
ITensor link and transpose from Julia theta-order storage into C++ storage.
The `D=1` fixture does not need this transpose because every virtual dimension
is one, so it is embedded directly in the C++ unit tests.

## Direction Inference

For the `3x2,D=2` fixtures:

- `h_link i;j->i;j+1` is east for site `(i,j)` and west for site `(i,j+1)`.
- `v_link i;j->i+1;j` is south for site `(i,j)` and north for site `(i+1,j)`.

The validator in `code/peps_cuda/julia_reference/validate_reference_fixtures.py`
does not need to assign directions for contraction; it contracts equal link
labels directly. A C++ importer will need the direction assignment above before
writing into `SiteTensor::{north,east,south,west}` order.
