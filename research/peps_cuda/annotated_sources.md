# Annotated Source Notes

This file is intentionally selective. It records sources that affect the CUDA
implementation strategy, not every adjacent paper that happened to be found.

## Core PEPS Algorithm

### Puente, Weerda, Schroeder, Rizzi 2025

- Local file: `sources/puente_weerder_schroeder_rizzi_2025_finite_peps.pdf`
- Link: https://arxiv.org/abs/2503.12557
- Role: primary source for finite-PEPS sampling, minSR, direct sampling, and the
  conceptual reason the optimization is sample-space rather than parameter-space.
- Useful implementation facts:
  - Keep `O` as sample rows, then solve in the `Ns x Ns` sample space.
  - Boundary-MPS contraction is the central cost; the CUDA implementation should
    optimize row absorption/reuse first.
  - Double-layer sampling environments may be stale/asynchronously refreshed,
    provided importance/sampling corrections are tracked.
- Viability: extremely high. This is the paper the thesis implementation should
  reproduce semantically.

### PEPS Structure Documentation

- Local file: `sources/PEPS_structure_documentation.pdf`
- Role: compact thesis-facing algorithm and workload document.
- Useful implementation facts:
  - Typical imaginary-time target: `8x8` to `32x32`, `D=2..8`, `d=2` most of
    the time, `Ns=1000..5000`.
  - Real-time target is smaller lattices around `8x8`, but much larger sample
    counts, often `Ns ~ Np`.
  - Direct sampler row formula maps naturally to sample-parallel GPU execution.
  - Single- and double-layer boundary-MPS cost formulas are the right first
    roofline inputs.
- Caution:
  - Sec. 3.3 appears to refer to the wrong equation number for the `Ns << Np`
    case. The described small system is the sample-space/minSR solve.
- Viability: high. This is the best local description of what code must expose.

### QuantumNaturalfPEPS.jl

- Local snapshot: `sources/QuantumNaturalfPEPS.jl-main/`
- Link: https://github.com/KonradSchroeder/QuantumNaturalfPEPS.jl
- Role: semantic reference implementation. Do not use its ITensor-heavy code as
  production structure.
- Useful implementation facts:
  - Stage order is `get_sample -> get_logpsi_and_envs ->
    get_all_horizontal_envs -> get_Ek -> get_Ok`.
  - `Ek.jl` buckets terms by changed-site geometry; CUDA should make this an
    explicit data layout.
  - `Ok.jl` writes zeros for unsampled physical sectors, which motivates a
    sparse/sliced `O` representation later.
- Viability: high as a correctness reference, low as a performance template.

### Vieijra, Haegeman, Verstraete, Vanderstraeten 2021

- Local file:
  `sources/vieijra_haegeman_verstraete_vanderstraeten_2021_direct_peps_sampling.pdf`
- Link: https://arxiv.org/abs/2109.07356
- Role: direct PEPS sampling precursor used by the Rizzi/Puente/Weerda/Schroeder
  paper.
- Useful implementation facts:
  - Direct sampling avoids Markov-chain autocorrelation by constructing
    conditional probabilities from approximate boundary contractions.
  - Importance correction is part of the method, not an optional afterthought.
- Viability: high for sampling-stage design, especially for explaining why the
  sample loop is sequential within a sample but parallel across samples.

### Chen and Heyl 2023/2024 MinSR

- Local file: `sources/chen_heyl_2023_minsr_neural_quantum_states.pdf`
- Link: https://arxiv.org/abs/2302.01941
- Role: source of the minimum-step stochastic reconfiguration idea used by the
  finite-PEPS paper.
- Useful implementation facts:
  - The important computational move is replacing the parameter-space SR matrix
    by a sample-space system when `Np >> Ns`.
  - For PEPS, this makes the solver memory manageable, but it does not remove
    the cost of generating `O` and `E`.
- Viability: high for solver rationale, low for contraction kernels because it
  is an NQS paper.

### Wu and Nys 2025/2026 PEPS-tVMC

- Local file: `sources/wu_nys_2026_peps_tvmc_gpu.pdf`
- Link: https://arxiv.org/abs/2512.06768
- Role: very relevant recent PEPS/tVMC reference with explicit GPU-scale
  claims.
- Useful implementation facts:
  - Reports `12x12`/`13x13` real-time PEPS runs on a single GPU card.
  - Uses minSR/TDVP ideas and emphasizes gauge redundancy, tensor locality, and
    Cholesky after conditioning.
  - Appendix/supplement text describes a "small-o trick": store sampled local
    sectors instead of full `O`, then reconstruct `O O^dagger`. This directly
    validates the compact sampled-sector path added to this scaffold.
- Viability: high for positioning and for the `O` memory strategy. The algorithm
  is not identical to the Rizzi/Schröder finite-PEPS paper, but the sampled
  sector memory trick is almost exactly aligned with the CUDA direction here.

### Chen, Jiang, Hangleiter, Schuch 2025 Sign Problem In TN Contraction

- Local file:
  `sources/chen_jiang_hangleiter_schuch_2025_sign_problem_tn_contraction.pdf`
- Link: https://arxiv.org/abs/2404.19023
- Role: contraction-complexity context that is also cited/discussed by the main
  finite-PEPS paper.
- Useful implementation facts:
  - Boundary-MPS contraction difficulty is tied to entanglement/correlation in
    the boundary state.
  - Positive/biased tensor structures can drastically reduce effective
    contraction hardness.
  - For PEPS expectation values, double-layer positivity helps explain favorable
    boundary-law behavior in important regimes.
- Viability: medium-high for thesis context and future gauge/preconditioning
  ideas. It does not directly prescribe CUDA kernels.

## Hardware And CUDA Documentation

### JUPITER Configuration And Technical Overview

- Links:
  - https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html
  - https://www.fz-juelich.de/en/jsc/jupiter/tech/
  - https://apps.fz-juelich.de/jsc/hps/jupiter/gpu-computing.html
  - https://apps.fz-juelich.de/jsc/hps/jupiter/affinity.html
- Role: target-node reality check.
- Useful implementation facts:
  - Booster node has 4x GH200 Grace-Hopper superchips.
  - Each GPU is Hopper/H100-class with 132 SMs, 96 GB HBM3, around 4 TB/s HBM.
  - CPU-GPU NVLink-C2C is 900 GB/s; GPU-GPU links are 150 GB/s per direction.
  - Slurm normally gives one GPU per task via `CUDA_VISIBLE_DEVICES`.
  - JUPITER's affinity docs warn that `srun` needs an explicit
    `--cpus-per-task` or `SRUN_CPUS_PER_TASK` to inherit the intended CPU count.
  - Nsight Compute may lock clocks to base values; compare profiler runs with
    that in mind.
- Viability: high. JUPITER is GH200/H100, not H200, based on current docs.

### NVIDIA Hopper Tuning Guide

- Local file: `sources/hopper_tuning_guide.pdf`
- Link: https://docs.nvidia.com/cuda/hopper-tuning-guide/
- Role: SM90 occupancy/memory model and Hopper-specific features.
- Useful implementation facts:
  - 64 resident warps/SM, 64K 32-bit registers/SM, 255 registers/thread.
  - Up to 228 KB shared memory/SM and 227 KB per block with opt-in.
  - TMA can move 1D-5D tensors between global and shared memory; useful only
    after the GEMM-backed boundary kernels reveal a stable hot shape.
- Viability: high for kernel design constraints.

### NVIDIA GPU Datasheets

- Local files:
  - `sources/nvidia_a100_datasheet.pdf`
  - `sources/nvidia_h100_datasheet.pdf`
  - `sources/nvidia_h200_datasheet.pdf`
- Role: hardware facts for the thesis hardware chapter and cost model.
- Useful implementation facts:
  - A100 40GB is much more memory constrained than the target GH200/H100 node.
  - H200 has the same headline FP64 compute as H100 SXM but larger/faster HBM.
  - JUPITER should still be modeled separately because its GH200 GPU has 96 GB
    HBM3 at about 4 TB/s, according to JUPITER docs.
- Viability: high for static specs; use live cluster measurements for actual
  clocks and achieved bandwidth.

### CUDA C++ Best Practices And Nsight Compute Guides

- Local files:
  - `sources/cuda_c_best_practices_guide.pdf`
  - `sources/nsight_compute_profiling_guide.pdf`
- Role: profiling and optimization checklist.
- Useful implementation facts:
  - Start with Nsight Systems to catch launch count, CPU gaps, copies, and stream
    overlap before tuning individual kernels.
  - Nsight Compute metrics should be attached to named kernel families:
    boundary absorption, sampling, `E`, `O`, and minSR.
- Viability: high, but generic.

### Simon Boehm CUDA Matmul Worklog

- Link: https://siboehm.com/articles/22/CUDA-MMM
- Role: mental model for memory coalescing, shared-memory tiling, register
  tiling, vectorized loads, autotuning, and warp tiling.
- Useful implementation facts:
  - Arithmetic intensity, not occupancy alone, decides whether a contraction is
    worth custom tiling.
  - For PEPS, this argues for cuBLASLt/grouped GEMM first, then custom kernels
    only for repeated small shapes where launch/library overhead dominates.
- Viability: medium-high. It is SGEMM-specific but very useful for intuition.

### cuBLAS Grouped GEMM Blog

- Link: https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
- Role: direct candidate for shape-bucketed PEPS contraction batches.
- Useful implementation facts:
  - Grouped GEMM supports variable matrix sizes, transposes, and scale factors in
    one launch.
  - cuBLASLt heuristic autotuning should be benchmarked on Ampere/Hopper instead
    of assuming default dispatch is optimal.
- Viability: high for the first cluster benchmarking pass.

### CUDA Graphs And cuBLASDx

- Links:
  - https://developer.nvidia.com/blog/cuda-graphs/
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
  - https://docs.nvidia.com/cuda/archive/13.0.1/cublasdx/index.html
- Role: later-stage launch-overhead and device-side BLAS options.
- Useful implementation facts:
  - CUDA Graphs reduce CPU launch overhead when a repeated workflow contains
    many short kernels or copies.
  - cuBLASDx enables selected BLAS operations inside CUDA kernels, which may be
    relevant for fusing very small fixed-shape contractions after library
    baselines are measured.
- Viability:
  - CUDA Graphs are likely useful once the sample-loop topology stabilizes.
  - cuBLASDx is a later optimization candidate, not a first implementation
    dependency.

### cuTENSOR And cuTensorNet

- Links:
  - https://developer.nvidia.com/cutensor
  - https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html
- Role: library alternatives for dense tensor contractions and general tensor
  network contraction planning.
- Useful implementation facts:
  - cuTENSOR supports direct tensor contractions, reductions, elementwise ops,
    arbitrary layouts, and mixed precision.
  - cuTensorNet provides contraction-path optimization, slicing under memory
    constraints, autotuning, and execution for general tensor networks.
- Viability:
  - cuTENSOR is worth benchmarking for awkward contraction shapes that do not map
    cleanly to GEMM without transposes.
  - cuTensorNet is likely less useful for the hot PEPS loop because finite-PEPS
    sampling repeatedly contracts networks with known row-MPS structure and
    truncation/reuse requirements. It is still useful as a reference/baseline for
    exact small contractions or path sanity checks.

### Other PEPS/Tensor-Contraction GPU Leads

- Examples found:
  - PEPSKit boundary-MPS documentation:
    https://quantumkithub.github.io/PEPSKit.jl/dev/examples/boundary_mps/
  - cuTENSORMg multi-GPU tensor contraction blog:
    https://developer.nvidia.com/blog/extending-block-cyclic-tensors-for-multi-gpu-with-nvidia-cutensormg/
  - GPU-accelerated TRG with PyTorch/CUDA: https://arxiv.org/abs/2306.00358
- Role: adjacent design context.
- Viability:
  - PEPSKit is useful for algorithm vocabulary, not CUDA code.
  - cuTENSORMg/cuTENSORMp are worth remembering for future multi-GPU dense
    contractions, but sample sharding is the simpler first multi-GPU strategy.
  - TRG/PyTorch GPU papers support the general thesis that tensor contractions
    and decompositions benefit from GPU libraries, but they do not solve the
    finite-PEPS sampling pipeline.

## Tensor-Network GPU State Of The Art

### Menczer et al. 2024 DGX-H100 DMRG

- Local file: `sources/menczer_legeza_2024_dgxh100_dmrg.pdf`
- Link: https://arxiv.org/abs/2407.07411
- Role: closest high-performance tensor-network GPU result in the source bundle.
- Useful implementation facts:
  - Reports 246 TFLOP/s sustained on a DGX-H100 node for DMRG.
  - Emphasizes hybrid CPU/multi-GPU execution and scheduling many tensor tasks.
- Transfer to PEPS: bucket by contraction shape and data locality; use the CPU
  for orchestration/term preprocessing, not per-contraction hot work.
- Viability: high for performance strategy, not a direct PEPS algorithm source.

### Menczer and Legeza 2023 Hybrid CPU-GPU TNS

- Local file: `sources/menczer_legeza_2023_hybrid_cpu_gpu_tns.pdf`
- Link: https://arxiv.org/abs/2305.05581
- Role: broader implementation framing for massively parallel tensor-network
  state algorithms on heterogeneous systems.
- Useful implementation facts:
  - Treats tensor-network work as a graph of many vector/matrix/tensor kernels
    with dependencies and reusable data.
  - Reinforces the need for task scheduling and shape-aware batching.
- Viability: medium-high for architecture design.

### Menczer and Legeza 2023 Non-Abelian Symmetry Follow-Up

- Local file: `sources/menczer_legeza_2023_nonabelian_tns_gpu.pdf`
- Link: https://arxiv.org/abs/2309.16724
- Role: shows how exploiting structure/symmetry can improve effective tensor
  network performance beyond raw kernel tuning.
- Useful implementation facts:
  - Reports order-of-magnitude complexity improvement and measured TFLOP/s gains
    over the earlier hybrid CPU-GPU baseline.
  - For finite PEPS, the analogous move is not necessarily non-Abelian symmetry
    immediately; it is exploiting operator geometry, physical-sector sparsity,
    stale boundary reuse, and sample/term bucketing.
- Viability: medium. Useful for thesis positioning and future extensions.

### GH200 Data Movement And BLAS Offload Papers

- Local files:
  - `sources/gh200_data_movement_2408.11556.pdf`
  - `sources/grace_hopper_blas_offload_2404.13195.pdf`
- Links:
  - https://arxiv.org/abs/2408.11556
  - https://arxiv.org/abs/2404.13195
- Role: placement guidance for Grace-Hopper systems.
- Useful implementation facts:
  - GH200 unified address space and NVLink-C2C are powerful, but memory placement
    still matters.
  - Automatic BLAS offload is encouraging for legacy codes, but this thesis
    should still keep hot PEPS data in HBM and call GPU libraries explicitly.
- Viability: medium-high for JUPITER/GH200 deployment decisions.

### Brower et al. 2025 Blackwell FP64 Emulation

- Local file: `sources/brower_legeza_2025_blackwell_fp64_emulation.pdf`
- Link: https://arxiv.org/abs/2510.04795
- Role: precision-policy background.
- Useful implementation facts:
  - Relevant to future mixed/emulated precision tensor-network methods.
  - Not a current implementation target because this project is A100 then
    GH200/H100/H200, not Blackwell.
- Viability: low for immediate code, useful as a "do not entangle precision
  policy with algorithm correctness" warning.
