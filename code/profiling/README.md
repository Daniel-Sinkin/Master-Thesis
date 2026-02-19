# A100 CUDA Profiling Warmups

## KPI pack (10 metrics)

Both SLURM scripts now profile a 10-KPI set (with per-metric fallbacks for Nsight version differences):

- `sm__throughput` `% of peak`
- `dram__throughput` `% of peak`
- `lts__...throughput` `% of peak` (L2 pressure proxy)
- `l1tex__t_sector_hit_rate.pct`
- `smsp__pipe_fma_cycles_active` `% of peak`
- `sm__warps_active` `% of peak`
- `launch__occupancy_limit_registers`
- `smsp__warps_eligible.*.per_cycle_active`
- `smsp__issue_active` `% of peak`
- `smsp__thread_inst_executed_per_inst_executed` (divergence-sensitive)

The scripts print the exact metric names they selected in the SLURM stdout log.

## Cases

- `gemm_fp32_good_cublas.cu`: cuBLAS FP32 GEMM with pedantic FP32 compute
- `gemm_fp32_bad_naive.cu`: naive FP32 GEMM kernel (intentionally inefficient)
- `warp_divergence_good_uniform.cu`: warp-uniform branch behavior (good)
- `warp_divergence_bad_divergent.cu`: forced intra-warp divergence (bad)

## Folder layout

- `code/profiling/CMakeLists.txt`: local build config for profiling binaries
- `code/profiling/ncu_fp32_profile.slurm`: one-shot compile + FP32 GEMM profile
- `code/profiling/ncu_warp_divergence_profile.slurm`: one-shot compile + divergence profile

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
