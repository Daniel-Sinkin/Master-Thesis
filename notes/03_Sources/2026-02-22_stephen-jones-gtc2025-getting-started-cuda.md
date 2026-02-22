---
id: 2026_02_22_stephen_jones_gtc2025_getting_started_cuda
aliases: []
tags:
  - source
  - cuda
  - programming-model
---

# Getting Started with CUDA and Parallel Programming (GTC 2025)

Citation: [YouTube talk](https://www.youtube.com/watch?v=GmNkYayuaA4)

## Claims (as stated)
- Most application code should not require hand-written GPU parallel programming.
- The practical CUDA stack is layered: frameworks first, then tuned libraries, then custom kernels only where necessary.
- Over-subscription is structurally important: launch more work than fits concurrently so the scheduler can keep SMs busy.
- Data-parallel kernel authoring is difficult and expensive to optimize; reusable primitives/libraries are usually better.

## Calibration (thesis-safe wording)
- "Do not write kernels" is a productivity heuristic, not an absolute rule. Custom kernels remain necessary for unsupported irregular workloads.
- Over-subscription improves utilization when work is sufficiently independent and resource limits are respected; it is not a guarantee of speedup.
- Early "cuTile" style claims are promising but should be treated as forward-looking unless reproduced for this thesis workloads.

## Relevance to thesis
- Directly supports a baseline ladder: cuBLAS/cuTENSOR first, custom kernels for irregular contraction paths where libraries miss.
- Supports keeping optimization effort concentrated on hotspot kernels rather than broad kernel rewrites.
- Supports workload decomposition into independent tasks for better scheduling and throughput.

## Actionable experiments
- Baseline ladder for each target kernel family:
  1) cuBLAS/cuTENSOR path
  2) hybrid path (library + custom pre/post kernels)
  3) fully custom path
- Measure productivity/performance trade-off:
  - time-to-first-correct result
  - time-to-80%-of-best throughput
  - best achieved throughput
- Over-subscription sweep:
  - vary grid depth and number of independent launches/streams
  - track throughput, occupancy, and launch overhead together

## Decision
- Adopt an abstraction-first workflow: highest available abstraction that meets performance and flexibility requirements.
- Reserve low-level custom kernel work for bottlenecks that remain after library/hybrid baselines.
