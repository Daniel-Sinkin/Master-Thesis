---
id: Optimization_Heuristics
aliases: []
tags: []
---

# Optimization Heuristics

## Rule
Store only what changes a decision, model, or experiment design.

## Current Heuristics
- Abstraction first: frameworks/libraries first, custom kernels only for remaining bottlenecks.
- Memory layout first.
- Occupancy/resource packing second.
- Concurrency when independence exists.

## Evidence Links
- [[../03_Sources/2026-02-22_stephen-jones-cuda-mental-model]]
- [[../03_Sources/2026-02-22_stephen-jones-gtc2025-getting-started-cuda]]

## Applied order for this thesis
1. Memory layout and coalescing
2. Occupancy and resource packing
3. Stream-level concurrency and oversubscription
