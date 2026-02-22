# Standalone CUDA experiments

This directory is the primary location for single-file `.cu` experiments that
are run through `code/profiling/orchestrate.py`.

## Optional sidecar config

For `my_experiment.cu`, you can add `my_experiment.json`.

Supported keys:

- `profile`: `none`, `nsys`, `ncu`, or `both`
- `libs`: list of CUDA libs, e.g. `["cudart", "cublas"]`
- `run_args`: list of CLI args passed to `main`
- `nvcc_flags`: extra NVCC flags
- `arch`: CUDA SM target, e.g. `80`
- `account`, `partition`, `time`, `cpus_per_task`, `gpus`: SLURM settings
- `nsys_trace`, `nsys_sample`: Nsight Systems options
- `ncu_metrics`: list of Nsight Compute metrics
- `ncu_kernel`: kernel regex for `ncu -k`
- `ncu_launch_skip`, `ncu_launch_count`: Nsight Compute launch control

CLI flags always override sidecar values.
