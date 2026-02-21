# A100 CUDA Profiling Warmups

## KPI pack (10 metrics)

Both SLURM scripts now profile a 10-KPI set (with per-metric fallbacks for Nsight version differences):

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__throughput.avg.pct_of_peak_sustained_elapsed` (L2 pressure proxy)
- `l1tex__t_sector_hit_rate.pct`
- `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `launch__occupancy_limit_registers`
- `smsp__warps_eligible.sum.per_cycle_active`
- `smsp__issue_active.avg.pct_of_peak_sustained_active`
- `smsp__thread_inst_executed_per_inst_executed.pct` (divergence-sensitive)

The scripts print the exact metric names they selected in the SLURM stdout log.

## Cases

- `gemm_fp32_good_cublas.cu`: cuBLAS FP32 GEMM with pedantic FP32 compute
- `gemm_fp32_bad_naive.cu`: naive FP32 GEMM kernel (intentionally inefficient)
- `warp_divergence_good_uniform.cu`: warp-uniform branch behavior (good)
- `warp_divergence_bad_divergent.cu`: forced intra-warp divergence (bad)
- `tn_two_site_good_batched_gemm.cu`: ordered two-step TN-style contraction via
  cuBLAS strided-batched GEMM
- `tn_two_site_bad_direct.cu`: direct two-site contraction kernel with poor
  contraction order and low reuse
- `tn_irregular_direct_bad_seq.cu`: irregular contraction path with repeated
  layout repack, one-launch-per-batch execution, and explicit sync/copies
- `tn_irregular_batched_good.cu`: same contraction family with coalesced access
  and one batched kernel launch per iteration

## Folder layout

- `code/profiling/CMakeLists.txt`: local build config for profiling binaries
- `code/profiling/ncu_fp32_profile.slurm`: one-shot compile + FP32 GEMM profile
- `code/profiling/ncu_warp_divergence_profile.slurm`: one-shot compile + divergence profile
- `code/profiling/ncu_tn_contraction_profile.slurm`: one-shot compile + TN
  contraction-order profile
- `code/profiling/nsight_irregular_training_profile.slurm`: combined Nsight
  Systems + Nsight Compute training pass for irregular contraction bottlenecks

## Build manually (from repository root)

```bash
module load Stages/2025 CUDA/12 GCC/13.3.0
cmake -S code/profiling -B code/profiling/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -DENABLE_CUDA_LINEINFO=ON -DFORCE_O3=ON
cmake --build code/profiling/build -j
```

## Run sanity checks manually

```bash
./code/profiling/build/gemm_fp32_good_cublas
./code/profiling/build/gemm_fp32_bad_naive
./code/profiling/build/warp_divergence_good_uniform
./code/profiling/build/warp_divergence_bad_divergent
```

## FP32 GEMM pair via SLURM

```bash
sbatch code/profiling/ncu_fp32_profile.slurm
```

Outputs:

- `code/profiling/ncu-fp32-out.<jobid>`
- `code/profiling/ncu-fp32-err.<jobid>`
- `code/profiling/good_fp32_metrics.<jobid>.csv`
- `code/profiling/bad_fp32_metrics.<jobid>.csv`

Expected behavior:

- `gemm_fp32_good_cublas`: much higher compute efficiency than naive.
- `gemm_fp32_bad_naive`: lower throughput and weaker scheduler efficiency.

## Warp divergence pair via SLURM

```bash
sbatch code/profiling/ncu_warp_divergence_profile.slurm
```

Outputs:

- `code/profiling/ncu-warpdiv-out.<jobid>`
- `code/profiling/ncu-warpdiv-err.<jobid>`
- `code/profiling/warp_uniform_metrics.<jobid>.csv`
- `code/profiling/warp_divergent_metrics.<jobid>.csv`

Expected behavior:

- `warp_divergence_good_uniform`: better thread participation and issue behavior.
- `warp_divergence_bad_divergent`: lower thread participation and lower effective throughput.

## TN two-site contraction pair via SLURM

```bash
sbatch code/profiling/ncu_tn_contraction_profile.slurm
```

Outputs:

- `code/profiling/ncu-tn-out.<jobid>`
- `code/profiling/ncu-tn-err.<jobid>`
- `code/profiling/good_tn_two_site_metrics.<jobid>.csv`
- `code/profiling/bad_tn_two_site_metrics.<jobid>.csv`

Expected behavior:

- `tn_two_site_good_batched_gemm`: high compute utilization from better
  contraction order + batched GEMM mapping.
- `tn_two_site_bad_direct`: lower throughput from direct $O(d^2\chi^2)$-style
  contraction with weak data reuse.

## Irregular contraction training (Nsight Systems + Compute)

```bash
sbatch code/profiling/nsight_irregular_training_profile.slurm
```

Outputs:

- `code/profiling/nsight-irregular-out.<jobid>`
- `code/profiling/nsight-irregular-err.<jobid>`
- `code/profiling/nsys_irregular_bad.<jobid>.nsys-rep`
- `code/profiling/nsys_irregular_good.<jobid>.nsys-rep`
- `code/profiling/bad_irregular_repack_metrics.<jobid>.csv`
- `code/profiling/bad_irregular_direct_metrics.<jobid>.csv`
- `code/profiling/good_irregular_batched_metrics.<jobid>.csv`

Training intent:

- Use Nsight Systems to confirm where end-to-end time is spent
  (kernel launch serialization, explicit synchronizations, D2H copies).
- Use Nsight Compute to attribute mechanism on selected kernels:
  repack kernel (memory/layout path), direct irregular kernel (serial-launch
  regime), and batched good kernel (coalesced/throughput-oriented regime).

## Thesis Phenomena Suite (7 benchmarks)

For a broader set of thesis-ready GPU phenomena plots (divergence sweep,
coalescing sweep, bank conflicts, launch overhead, ILP/dependency, register
pressure, arithmetic intensity), use:

```bash
bash code/profiling/phenomena/submit_all_phenomena.sh
```

See:

- `code/profiling/phenomena/README.md`
- `code/profiling/phenomena/slurm/`
- `code/profiling/phenomena/results/`
