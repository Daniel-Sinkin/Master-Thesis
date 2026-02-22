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

## Decisions Log
- [x] Optimization order: layout/coalescing -> occupancy/resource packing -> concurrency
- [x] 128 threads per block treated as baseline heuristic, not hard rule

## Linked Notes
- [[../02_Concepts/Optimization_Heuristics]]
- [[../03_Sources/Source_Index]]
- [[../03_Sources/2026-02-22_stephen-jones-cuda-mental-model]]
