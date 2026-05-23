# Source Inventory

Local source bundle: `research/peps_cuda/sources/`

This inventory is deliberately small. The fuller comments are in
`annotated_sources.md`.

## Core PEPS

| File | Approx size | Use |
| --- | ---: | --- |
| `PEPS_structure_documentation.pdf` | 456 KiB | Local 12-page procedure/workload notes. |
| `puente_weerder_schroeder_rizzi_2025_finite_peps.pdf` | 1.4 MiB | Main paper: finite PEPS, sampling, minSR, contraction complexity. |
| `puente_weerder_schroeder_rizzi_2025_finite_peps_source.tar.gz` | 2.2 MiB | Paper source, used for precise equation/algorithm anchors. |
| `main.tex`, `main.bbl`, `literature.bib`, `images/` | extracted | Extracted arXiv source and figures. |
| `QuantumNaturalfPEPS.jl-main.zip` | 48 KiB | Julia reference snapshot from Konrad Schröder repo. |
| `QuantumNaturalfPEPS.jl-main/` | 224 KiB | Extracted semantic reference implementation. |

## Sampling And minSR

| File | Approx size | Use |
| --- | ---: | --- |
| `vieijra_haegeman_verstraete_vanderstraeten_2021_direct_peps_sampling.pdf` | 1.1 MiB | Direct PEPS sampling precursor. |
| `chen_heyl_2023_minsr_neural_quantum_states.pdf` | 3.8 MiB | minSR background and sample-space SR motivation. |
| `wu_nys_2026_peps_tvmc_gpu.pdf` | 1.0 MiB | Recent PEPS-tVMC/single-GPU context; validates sampled-sector `O` memory strategy. |
| `chen_jiang_hangleiter_schuch_2025_sign_problem_tn_contraction.pdf` | 1.5 MiB | Contraction-complexity/sign-structure context for boundary-MPS hardness. |

## NVIDIA/CUDA/Hopper

| File | Approx size | Use |
| --- | ---: | --- |
| `cuda_c_best_practices_guide.pdf` | 2.2 MiB | Generic CUDA optimization and memory-practice reference. |
| `hopper_tuning_guide.pdf` | 168 KiB | SM90/Hopper occupancy, shared memory, TMA, cluster notes. |
| `nsight_compute_profiling_guide.pdf` | 2.9 MiB | Metric definitions and profiler behavior. |
| `nvidia_a100_datasheet.pdf` | 484 KiB | A100 baseline facts. |
| `nvidia_h100_datasheet.pdf` | 772 KiB | H100 baseline facts. |
| `nvidia_h200_datasheet.pdf` | 628 KiB | H200 cloud-target facts. |

## Tensor-Network GPU / GH200

| File | Approx size | Use |
| --- | ---: | --- |
| `menczer_legeza_2023_hybrid_cpu_gpu_tns.pdf` | 612 KiB | Hybrid CPU-GPU tensor-network architecture. |
| `menczer_legeza_2023_nonabelian_tns_gpu.pdf` | 548 KiB | Structure/symmetry performance lesson. |
| `menczer_legeza_2024_dgxh100_dmrg.pdf` | 268 KiB | DGX-H100 DMRG performance reference. |
| `gh200_data_movement_2408.11556.pdf` | 4.2 MiB | Grace-Hopper data placement and movement. |
| `grace_hopper_blas_offload_2404.13195.pdf` | 124 KiB | GH200 BLAS/offload placement context. |
| `brower_legeza_2025_blackwell_fp64_emulation.pdf` | 2.5 MiB | Future precision-policy context, not current target. |
| `ozaki_scheme_ii_2504.08009.pdf` | 2.1 MiB | Ozaki-II FP64 emulation using modular/CRT GEMM. |
| `mukunoki_2025_dgemm_without_fp64_ozaki_fp8.pdf` | 640 KiB | FP8/Blackwell Ozaki DGEMM context. |

## External Links Rechecked 2026-05-15

These were opened again during the timed implementation/research pass. Recheck
them once more before final thesis citation, because JUPITER early-access
details and arXiv versions can still move.

- Main paper: https://arxiv.org/abs/2503.12557
- PEPS-tVMC single-GPU context: https://arxiv.org/abs/2512.06768
- Sign problem in tensor-network contraction: https://arxiv.org/abs/2404.19023
- Julia repo: https://github.com/KonradSchroeder/QuantumNaturalfPEPS.jl
- QuantumNaturalGradient repo:
  https://github.com/NeTeNeSyQuMa/QuantumNaturalGradient.jl
- JUPITER configuration:
  https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html
- JUPITER GPU computing:
  https://apps.fz-juelich.de/jsc/hps/jupiter/gpu-computing.html
- Hopper tuning guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/
- CUDA C++ best practices:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Nsight Compute docs:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- cuBLAS grouped GEMM:
  https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
- cuBLASDx:
  https://docs.nvidia.com/cuda/cublasdx/0.5.1/api/index.html
- CUTLASS/CuTe:
  https://docs.nvidia.com/cutlass/media/docs/cpp/cute/index.html
- Simon Boehm CUDA matmul:
  https://siboehm.com/articles/22/CUDA-MMM
- Ozaki-II:
  https://arxiv.org/abs/2504.08009
- FP8 Ozaki DGEMM:
  https://arxiv.org/abs/2508.00441
