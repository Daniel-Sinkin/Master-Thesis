# GPU Phenomena Microbenchmark Suite

This folder contains CUDA microbenchmarks for thesis-ready plots, plus
SLURM scripts and a dependency-based submitter.

## Benchmarks

1. `phenomenon_divergence_sweep`:
   sweeps fraction of divergent warps from 0 to 1.
2. `phenomenon_coalescing_sweep`:
   sweeps global-memory access stride.
3. `phenomenon_shared_bank_conflict`:
   compares shared-memory transpose with/without padding.
4. `phenomenon_launch_overhead`:
   compares many tiny launches vs one fused launch.
5. `phenomenon_ilp_dependency`:
   compares dependent instruction chains vs independent ILP.
6. `phenomenon_register_pressure`:
   compares low vs high register pressure and reports theoretical occupancy.
7. `phenomenon_arithmetic_intensity_sweep`:
   sweeps compute-per-byte to create roofline-style curves.
8. `phenomenon_hbm_stride_interval_raw`:
   raw-run HBM stride benchmark (8-byte reads, fixed stride-byte list).
9. `phenomenon_gemm_value_switching`:
   SGEMM sweep over data-initialization patterns for value-dependent power tests.

## Build once

From repository root:

```bash
module load Stages/2025 CUDA/12 GCC/13.3.0
cmake -S code/profiling -B code/profiling/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 -DENABLE_CUDA_LINEINFO=ON -DFORCE_O3=ON
cmake --build code/profiling/build -j
```

## Run all sequentially via SLURM dependencies

From repository root:

```bash
bash code/profiling/phenomena/submit_all_phenomena.sh
```

Optional mode:

- `afterok` (default): next job starts only if previous succeeds
- `afterany`: next job starts regardless of previous job state

```bash
bash code/profiling/phenomena/submit_all_phenomena.sh afterany
```

## Scripts and outputs

SLURM scripts are under `code/profiling/phenomena/slurm`.
Each run writes CSV output under `code/profiling/phenomena/results`.

Expected CSV files:

- `01_divergence_sweep.<jobid>.csv`
- `02_coalescing_sweep.<jobid>.csv`
- `03_shared_bank_conflict.<jobid>.csv`
- `04_launch_overhead.<jobid>.csv`
- `05_ilp_dependency.<jobid>.csv`
- `06_register_pressure.<jobid>.csv`
- `07_arithmetic_intensity.<jobid>.csv`
- `08_hbm_stride_raw.csv`
- `08_hbm_stride_ncu_summary.csv`
- `09_gemm_value_switching_raw.csv`

For the stride-vs-bandwidth experiment with Nsight Compute overlay, submit:

```bash
sbatch code/profiling/phenomena/slurm/08_hbm_stride_raw_with_ncu.slurm
```

For the GEMM value-pattern energy experiment, submit:

```bash
sbatch code/profiling/phenomena/slurm/09_gemm_value_switching_energy.slurm
```

Default run controls for this job:

- `M=4096 N=4096 K=4096`
- `WARMUP=50`
- `ITERS=1000` (used as calibration seed)
- `TARGET_TIMED_SECONDS=120` (effective timed window per mode)
- `REPEATS=1`
- `POWER_SAMPLE_MS=50`

After the job completes:

```bash
python3 scripts/plot_hbm_stride_interval_raw.py \
  --input code/profiling/phenomena/results/08_hbm_stride_raw.csv \
  --ncu-summary code/profiling/phenomena/results/08_hbm_stride_ncu_summary.csv \
  --output-median images/hbm_stride_interval_requested_median.pdf \
  --output-ncu images/hbm_stride_interval_ncu_dram.pdf
```

## Generate thesis figures

From repository root:

```bash
python3 -m pip install matplotlib
python3 scripts/generate_phenomena_plots.py
```

The script reads the newest matching CSV per benchmark from
`code/profiling/phenomena/results/` and writes PDF plots to `images/`.
