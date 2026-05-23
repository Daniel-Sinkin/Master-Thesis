# Benchmark Triage Snapshot

Generated with:

```bash
python3 code/peps_cuda/tools/benchmark_matrix.py --gpu jupiter_gh200 \
  --lattices 4x4,8x8,16x16,32x32 \
  --d-values 2,4,6,8 \
  --dc-values 16,32,64,96,128 \
  --samples 128,512,2000,5000
```

The current triage model is intentionally simple: it treats dense `O` and
sampled-sector `O` storage as the limiting first-order HBM pressure. It does
not include cuBLAS/cuSOLVER workspaces, allocator padding, cached environments,
or profiler overhead, so anything close to the limit should be treated as
unsafe.

## Key Transition Points On One 96 GiB GH200 GPU

| Case | Dense `O` | Sampled-sector `O` | Triage |
| --- | ---: | ---: | --- |
| `8x8,D=8,Ns=5000` | `39.06 GiB` | `19.53 GiB` | dense debug run plausible |
| `16x16,D=6,Ns=5000` | `49.44 GiB` | `24.72 GiB` | dense debug run plausible |
| `16x16,D=8,Ns=2000` | `62.50 GiB` | `31.25 GiB` | dense debug run plausible but tight |
| `16x16,D=8,Ns=5000` | `156.25 GiB` | `78.12 GiB` | direct Gram strongly preferred |
| `32x32,D=6,Ns=2000` | `79.10 GiB` | `39.55 GiB` | sampled-sector only |
| `32x32,D=6,Ns=5000` | `197.75 GiB` | `98.88 GiB` | direct Gram required |
| `32x32,D=8,Ns=512` | `64.00 GiB` | `32.00 GiB` | dense debug run plausible but tight |
| `32x32,D=8,Ns=2000` | `250.00 GiB` | `125.00 GiB` | direct Gram required |
| `32x32,D=8,Ns=5000` | `625.00 GiB` | `312.50 GiB` | direct Gram required |

## Consequence

The staged implementation should still support materialized dense and compact
`O` because those are excellent for debugging and A100/H100 profiler bring-up.
But the thesis-scale path must be:

1. compute row/site gradient slices,
2. immediately contribute them to the sample-space Gram and `O^dagger x`,
3. discard or checkpoint only compressed diagnostics,
4. shard samples across GPUs before trying to push `Ns=5000`.

This is independent of boundary-MPS contraction speed. If `O` storage is
wrong, faster contractions only reach the memory wall faster.
