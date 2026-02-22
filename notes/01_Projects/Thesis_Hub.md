---
id: Thesis_Hub
aliases: []
tags: []
---

# Thesis Hub

## Active Questions
- [ ]
- [ ]

## Current Experiments
- [ ] Block-size sweep (64/128/256/512) on irregular kernels
- [ ] Stride/coalescing sweep for irregular index mappings
- [ ] Single-stream vs multi-stream oversubscription comparison
- [ ] Baseline ladder per kernel: cuBLAS/cuTENSOR -> hybrid -> fully custom

## Decisions Log
- [x] Optimization order: layout/coalescing -> occupancy/resource packing -> concurrency
- [x] 128 threads per block treated as baseline heuristic, not hard rule
- [x] Abstraction-first workflow: use highest-level CUDA abstraction that meets needs

## Linked Notes
- [[../02_Concepts/Optimization_Heuristics]]
- [[../03_Sources/Source_Index]]
- [[../03_Sources/2026-02-22_stephen-jones-cuda-mental-model]]
- [[../03_Sources/2026-02-22_stephen-jones-gtc2025-getting-started-cuda]]
