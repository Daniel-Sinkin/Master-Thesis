# A100 FP32 Profiling Warmup: One KPI, Two Cases

## KPI to focus on

Use this single KPI in Nsight Compute:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`

Interpretation:

- higher = closer to the GPU's sustained SM throughput peak
- lower = further from peak

## Cases

- `gemm_fp32_good_cublas.cu`: cuBLAS FP32 GEMM with pedantic FP32 compute
- `gemm_fp32_bad_naive.cu`: naive FP32 GEMM kernel (intentionally inefficient)

## Folder layout

- `code/profiling/CMakeLists.txt`: local build config for profiling binaries
- `code/profiling/ncu_fp32_profile.slurm`: one-shot compile + profile batch script

## Build manually (from repository root)

```bash
module load Stages/2025 CUDA/12 GCC/13.3.0
cmake -S code/profiling -B code/profiling/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -DENABLE_CUDA_LINEINFO=ON -DFORCE_O3=ON
cmake --build code/profiling/build -j
```

## Run manually (throughput sanity)

```bash
./code/profiling/build/gemm_fp32_good_cublas
./code/profiling/build/gemm_fp32_bad_naive
```

Default sizes:

- good: `8192 x 8192 x 8192`, warmup 20, iters 100
- bad: `4096 x 4096 x 4096`, warmup 5, iters 20

## Profile manually (Nsight Compute)

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed --launch-skip 20 --launch-count 1 ./code/profiling/build/gemm_fp32_good_cublas
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -k naive_sgemm_fp32 --launch-skip 5 --launch-count 1 ./code/profiling/build/gemm_fp32_bad_naive
```

If this metric name differs in your Nsight version:

```bash
ncu --query-metrics | grep -Ei "sm__throughput.*pct_of_peak"
```

## One-command SLURM path

Submit from repository root:

```bash
sbatch code/profiling/ncu_fp32_profile.slurm
```

Outputs:

- `code/profiling/ncu-fp32-out.<jobid>`
- `code/profiling/ncu-fp32-err.<jobid>`
- `code/profiling/good_fp32_tpp.<jobid>.csv`
- `code/profiling/bad_fp32_tpp.<jobid>.csv`

## Expected behavior

- `gemm_fp32_good_cublas`: clearly higher `% of peak`, often around or above 50% on healthy A100 FP32 runs.
- `gemm_fp32_bad_naive`: much lower `% of peak`.
