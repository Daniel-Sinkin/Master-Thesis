---
id: 2026_02_22_stephen_jones_cuda_mental_model
aliases: []
tags:
  - source
  - cuda
  - profiling
---

# Stephen Jones CUDA talk (2026-02-22)

Citation: [YouTube talk](https://www.youtube.com/watch?v=QQceTDjA4f4)

## Claims (as stated)
- Memory access patterns dominate performance and can produce order-of-magnitude differences.
- Occupancy and resource packing are the second major performance lever.
- Concurrency and oversubscription give the runtime/scheduler more freedom to improve utilization.
- Memory bandwidth is often the primary limiting factor on modern GPUs.

## Calibration (thesis-safe wording)
- "Never below 128 threads per block" is a practical starting heuristic for many throughput-oriented, bandwidth-sensitive kernels, not a universal rule.
- "128 threads x 8 bytes = 1024 bytes page" is an intuition from a specific access pattern and benchmark context, not a universal DRAM page law.
- "Bandwidth is the primary limiter" is true for many kernels in this thesis domain, but compute-bound regimes still appear at sufficient arithmetic intensity.

## Relevance to thesis
- Supports memory-layout-first optimization strategy for irregular contraction kernels.
- Supports explicit occupancy/resource discussion and block-size sweeps.
- Supports stream-level concurrency experiments in the profiling workflow.
- Maps directly to Nsight KPI interpretation (memory throughput, occupancy, issue efficiency).

## Actionable experiments
- Block-size sweep at fixed shape: 64 / 128 / 256 / 512 threads per block.
- Stride/coalescing sweep across irregular index mappings.
- Oversubscription test: single stream vs multi-stream independent work.

## Decision
- Keep optimization order: memory layout/coalescing first, occupancy/resource packing second, concurrency third.
- Treat 128-thread blocks as a tuning baseline, then verify with Nsight Compute and runtime measurements.
