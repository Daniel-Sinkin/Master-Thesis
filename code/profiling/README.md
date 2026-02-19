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

## Build

```bash
nvcc -O3 -arch=sm_80 /Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_good_cublas.cu -lcublas -o /Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_good_cublas
nvcc -O3 -arch=sm_80 /Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_bad_naive.cu -o /Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_bad_naive
```

## Quick run (throughput sanity)

```bash
/Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_good_cublas
/Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_bad_naive
```

Defaults:

- good: `8192 x 8192 x 8192`, warmup 20, iters 100
- bad: `4096 x 4096 x 4096`, warmup 5, iters 20

## Nsight Compute

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -k regex:.*gemm.* --launch-skip 20 --launch-count 1 /Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_good_cublas
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -k naive_sgemm_fp32 --launch-skip 5 --launch-count 1 /Users/danielsinkin/GitHub_private/Master-Thesis/code/profiling/gemm_fp32_bad_naive
```

If this metric name differs in your Nsight version:

```bash
ncu --query-metrics | rg "sm__throughput.*pct_of_peak"
```

## Expected behavior

- `gemm_fp32_good_cublas`: clearly higher `% of peak`, often around or above 50% on healthy A100 FP32 runs.
- `gemm_fp32_bad_naive`: much lower `% of peak`.
