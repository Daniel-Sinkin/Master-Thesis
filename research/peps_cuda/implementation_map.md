# Implementation Map: Julia Reference To C++/CUDA Scaffold

The goal is semantic compatibility with `QuantumNaturalfPEPS.jl`, not a literal
translation of ITensor objects.

| Julia reference stage | Current scaffold | Production CUDA direction |
| --- | --- | --- |
| `PEPS(...)` tensor storage | `PEPS`, `SiteTensor`, `PackedPEPS` in `peps.hpp` | Keep physical-major packed slices; add optional padded interior layout for GEMM buckets. |
| `get_sample` | `sample_exact_small` and cached exact batch distribution | Appendix-B direct sampler with stale/reused double-layer boundary-MPS, sample-parallel workers, GPU RNG. |
| `generate_double_layer_envs` | documented only | Double-layer boundary-MPS row absorption with small `Dc_double`, refreshed asynchronously on its own stream. |
| `get_logψ_and_envs` | `contract_amplitude_exact` | Single-layer boundary-MPS top/down environments with `Dc`, cuBLASLt/grouped GEMM, compression. |
| `get_all_horizontal_envs` | exact contraction indirectly recomputes amplitudes | Row left/right environments for each sampled row, reused by `E` and `O`. |
| `get_precomp_sOψ_elems` | `enumerate_flip_contributions` | CPU precompute/expand Hamiltonian terms; transfer compact flip records by bucket. |
| `Ek.sort_dict` | `classify_flip_sites`, `summarize_flip_buckets` | Separate GPU kernels/GEMM buckets for diagonal, horizontal, vertical, plaquette, long-horizontal, fallback. |
| `get_Ek` | `local_energy_exact` | Reuse vertical/horizontal/plaquette environments; diagonal terms stay scalar. |
| `get_Ok` | `log_gradients_exact`, compact sampled-sector rows | Compute sampled physical sector per site; avoid materializing zero physical sectors. |
| `compute_importance_weights` | `compute_importance_weights`, CUDA normalized weight kernel | Compute log-ratios on GPU, reduce `logsumexp`, normalize to mean one. |
| `QuantumNaturalGradient.evolve` / SR/minSR | `sr_direction_parameter_space`, `minsr_direction_weighted`, sampled-sector minSR direction, dense and sampled-sector CUDA Gram/apply helper kernels | Use sample-space minSR for large PEPS; keep parameter-space SR as tiny/debug or real-time-small-`Np` fallback. |

## Current C++/CUDA Files

- `include/peps_cuda/peps.hpp`: host data model and CPU semantic API.
- `include/peps_cuda/cuda_kernels.hpp`: CUDA launch API for first kernels.
- `include/peps_cuda/memory.hpp`: process RSS/peak memory helpers for local
  and cluster lifetime traces.
- `src/peps_cpu.cpp`: exact tiny-system CPU oracle, Hamiltonian expansion,
  importance weights, minSR, packed layout.
- `src/peps_cuda.cu`: first data-parallel CUDA kernels.
- `src/gpu_smoke.cu`: CUDA smoke test for cluster/Nsight runs.
- `src/tests.cpp`: CPU unit tests, including embedded Julia `D=1`, real `D=2`,
  and complex `D=2` fixture checks.
- `julia_reference/export_reference_fixtures.jl`: Julia-side fixture exporter
  with ITensor axis labels, row-major sample copy, and memory facts.
- `julia_reference/validate_reference_fixtures.py`: independent fixture
  validator that reconstructs small PEPS amplitudes and `O_k` rows from the
  exported JSONL.
- `tools/estimate_peps_costs.py`: boundary-MPS cost and dense-`O` memory model.
- `tools/benchmark_matrix.py`: benchmark-case CSV with dense/sampled/direct
  Gram memory triage.
- `tools/memory_pressure.py`: precision-dependent FP64/FP32/FP16-storage
  working-set model.
- `tools/occupancy_scratch.py`: A100/H100 occupancy scratch estimates.
- `tools/run_cpu_regression.sh`: local no-CUDA regression wrapper.

## Highest-Risk Missing Pieces

1. Boundary-MPS compression:
   - Need a concrete SVD/density-matrix compression implementation.
   - Need tests showing convergence to exact contraction as `Dc` increases.
2. Direct sampling:
   - Need GPU-side row conditional probabilities.
   - Need stable RNG and log-probability accumulation.
3. Sparse/sliced `O`:
   - Sampled-sector compaction and Gram smoke kernels exist.
   - Direct boundary-environment Gram accumulation is still needed for
     production `D=8`, `32x32` workloads.
4. Library benchmarks:
   - Need compare cuBLAS loop, strided batched, grouped GEMM, cuTENSOR, and
     custom kernels on actual A100/H100/GH200.
5. Multi-GPU:
   - First split samples across GPUs.
   - Reduce sample-space Gram and force vector across ranks.
   - Avoid splitting individual contractions until sample sharding stops scaling.
