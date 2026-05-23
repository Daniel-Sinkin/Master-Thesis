# Tensor-Network GPU SOTA Brief

This is a working synthesis for thesis/project planning. It is deliberately
implementation-oriented: what matters for this PEPS CUDA project, what does not,
and what should be benchmarked before hand-writing kernels.

## High-Level Read

The strongest nearby GPU tensor-network results are not PEPS sampling codes.
They are DMRG/TNS codes that succeed by decomposing tensor-network algorithms
into many structured dense linear algebra tasks, batching/scheduling them across
CPU and GPUs, and exploiting problem structure. That is still highly relevant:
finite-PEPS sampling has the same basic performance trap, namely many small or
medium tensor contractions with shape variation and reusable environments.

The practical lesson is:

- Use the PEPS row/boundary structure explicitly.
- Bucket by tensor shape and operator geometry.
- Use cuBLASLt/grouped GEMM/cuTENSOR before custom kernels.
- Keep CPU work at orchestration/preprocessing granularity, not per contraction.
- Avoid dense materialization of `O` except as a debug baseline.

Useful primary links:

- Puente/Weerda/Schröder/Rizzi finite PEPS:
  https://arxiv.org/abs/2503.12557
- QuantumNaturalfPEPS.jl:
  https://github.com/KonradSchroeder/QuantumNaturalfPEPS.jl
- JUPITER configuration:
  https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html
- NVIDIA Hopper tuning guide:
  https://docs.nvidia.com/cuda/hopper-tuning-guide/
- cuBLAS grouped GEMM:
  https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
- Simon Boehm CUDA matmul worklog:
  https://siboehm.com/articles/22/CUDA-MMM
- Menczer/Legeza hybrid CPU-GPU TNS:
  https://arxiv.org/abs/2305.05581
- Wu/Nys PEPS-tVMC:
  https://arxiv.org/abs/2512.06768
- Menczer/Legeza DGX-H100 DMRG:
  https://arxiv.org/abs/2407.07411
- Brower/Legeza Blackwell FP64 emulation:
  https://arxiv.org/abs/2510.04795

## Core PEPS-Specific Sources

Puente, Weerda, Schröder, and Rizzi (2025) is the primary algorithm source. It
combines finite PEPS, direct sampling, stale/reused double-layer environments,
and minSR. Its important computational move is solving the stochastic
reconfiguration problem in sample space, but that only moves the linear solve;
it does not make `O` and `E` cheap.

Vieijra, Haegeman, Verstraete, and Vanderstraeten (2021) is the direct-sampling
precursor. It explains the direct conditional sampling logic that makes samples
independent but leaves sequential dependence inside each sample. For GPU code,
this means the first parallel axis is the sample batch, not columns within one
sample.

QuantumNaturalfPEPS.jl is a semantic reference, not a performance reference.
The useful parts to mirror are the stage ordering and the `E` term bucketing
logic, not the ITensor object model or allocation behavior.

## GPU Tensor-Network Work To Learn From

Menczer and Legeza (2023) frame high-performance tensor-network algorithms as a
hybrid CPU-GPU task graph. The transfer to PEPS is shape-aware batching: many
small contractions should be grouped into fewer high-throughput library calls or
fused kernels.

Menczer et al. (2024) report a DGX-H100 DMRG result at the scale of hundreds of
TFLOP/s sustained. That does not provide a PEPS sampler, but it demonstrates
that tensor-network algorithms can use H100-class hardware well when their
linear algebra is batched and scheduled rather than dispatched as isolated tiny
calls.

Brower et al. (2025) apply mixed-precision/Ozaki-style FP64 emulation to
ab-initio tensor-network state calculations for Blackwell-era hardware. The
important lesson for this project is architectural separation: keep precision
policy behind a backend boundary. The immediate A100/Hopper PEPS code should
not depend on Ozaki, but the GEMM backend should be swappable enough that a
future Blackwell or weak-FP64 experiment can be added without rewriting PEPS
sampling logic.

The non-Abelian symmetry follow-up is useful for a broader lesson: algorithmic
structure can beat raw kernel work. For this PEPS thesis, the first equivalent
structures are not non-Abelian blocks; they are sampled physical sectors,
operator-support buckets, row environments, stale double-layer reuse, and direct
sample-space Gram accumulation.

Wu and Nys (2025/2026) is the closest newly found PEPS/VMC GPU-adjacent source.
It reports real-time PEPS-tVMC simulations on `12x12`/`13x13` lattices using a
single GPU card and explicitly discusses a "small-o" memory trick: construct the
minSR matrix from sampled local sectors instead of materializing the full `O`.
That is essentially the same memory move as the compact sampled-sector path in
this scaffold, so it is a strong external validation of the direction.

## NVIDIA Library Baselines

cuBLAS grouped GEMM is a high-priority benchmark because PEPS boundary
contractions naturally produce groups of related but not identical GEMMs. It
supports variable shapes/transposes/scales in one grouped launch, which attacks
the launch-overhead problem without immediately writing custom kernels.

cuBLASLt should be benchmarked for dense Gram and larger contraction buckets
because its heuristics and autotuning can choose Hopper/Ampere-specific
algorithms. For early correctness, the custom Gram kernels in the scaffold are
fine; for real `Ns=1000..5000`, library GEMM should be the baseline.

cuTENSOR is worth testing for contractions that otherwise require awkward
transposes or many small GEMMs. Its JIT and plan cache are especially relevant
if the same contraction descriptor repeats across many samples or optimization
steps.

cuTensorNet is less likely to be the hot-loop solution for this project because
the PEPS algorithm depends on boundary-MPS truncation and environment reuse, not
just one-shot exact contraction path optimization. It remains useful as an exact
small-network baseline and sanity check.

## Hopper/GH200 Implications

JUPITER Booster nodes should be treated as four GH200/Hopper GPUs per node with
96 GB HBM3 each, not as H200 141 GB GPUs. The 96 GB capacity is large enough for
some dense debug runs, but not enough for production dense `O` at `D=8`,
`32x32`, and thousands of samples.

Hopper features to remember:

- 132 SMs on the JUPITER GPU configuration.
- 96 GB HBM3 at about 4 TB/s on JUPITER.
- 50 MB L2 on H100-class GPUs.
- 228 KB shared memory per SM and up to 227 KB per block with opt-in.
- TMA and thread-block clusters are promising only after stable hot shapes are
  identified.

The immediate GH200 rule is simple: hot tensors and environments belong in HBM.
Grace memory and NVLink-C2C are excellent for staging/orchestration and maybe
overflow experiments, but relying on Grace memory for hot contractions will
usually turn a compute problem into a data-movement problem.

## What This Means For The PEPS Implementation

### Sampling

The direct sampler should keep a refreshed double-layer environment version and
spawn many sample workers. Within a sample, rows/sites remain conditionally
dependent. Across samples, the GPU can run large batches.

Implementation target:

- Precompute or refresh double-layer boundaries asynchronously.
- For each sample batch, project physical slices and update sampled boundaries.
- Track `logpc` so importance weights can correct stale approximate sampling.

### E Splitting

The Hamiltonian should be expanded into changed-site records and bucketed by
geometry:

- diagonal,
- single-site,
- horizontal nearest,
- vertical nearest,
- plaquette,
- horizontal long,
- fallback.

Diagonal terms should never invoke a PEPS contraction. Horizontal and plaquette
terms should reuse horizontal/vertical environments.

### O Splitting

Dense `O` is a debug baseline. Sampled-sector `O` is the first meaningful memory
reduction. Direct Gram accumulation is the production direction.

Current scaffold status:

- Dense CPU minSR.
- Weighted dense CPU minSR.
- Dense CUDA smoke Gram.
- Compact sampled-sector CPU Gram and minSR direction.
- Compact sampled-sector CUDA smoke Gram.

Next production step:

- Replace compact rows built from exact dense gradients with compact rows built
  directly from boundary environments.
- Accumulate Gram by site/environment slices without materializing the full
  compact `Ns x sampled_parameters` matrix when it does not fit.

## Profiler-Driven Optimization Order

1. Nsight Systems: verify launch count, CPU gaps, host-device copies, stream
   overlap, and allocation behavior.
2. cuBLAS/cuBLASLt/cuTENSOR baselines for shape buckets.
3. Nsight Compute on one kernel family at a time.
4. CUDA Graphs once the loop topology is stable.
5. CuTe/CUTLASS/cuBLASDx custom kernels only for repeated shapes that libraries
   leave underutilized.

## Precision Policy

Start with complex FP64 for correctness and comparisons. Then test mixed
precision only behind explicit accuracy gates:

- FP32 tensor storage with FP64 accumulation.
- TF32/BF16 only for contractions where sampling/compression noise dominates.
- Compare energy and update direction against FP64 on small systems.

Blackwell FP64 emulation papers are useful background, especially for future
hardware discussions, but this project targets A100 and Hopper/GH200/H100/H200.
Do not let Blackwell precision strategy complicate the first implementation.

## Open SOTA Questions

- Does cuTENSOR beat grouped GEMM for any boundary-MPS contraction shape after
  accounting for plan/cache overhead?
- Can the row absorption and compression be organized into large enough GEMM
  groups to avoid custom kernels for `D<=8`?
- Does sampled-sector direct Gram become memory-bound before compute-bound on
  GH200?
- At what `Ns` and lattice size does sample sharding across four GPUs stop
  scaling, forcing a more complicated split of contractions/environments?
- How stale can double-layer environments be on the target models before
  importance weights become noisy enough to erase the speedup?
