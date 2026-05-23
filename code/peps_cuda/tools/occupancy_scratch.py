#!/usr/bin/env python3
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    name: str
    max_threads_per_sm: int
    max_blocks_per_sm: int
    max_warps_per_sm: int
    regs_per_sm: int
    smem_per_sm: int
    reg_alloc_granularity_per_warp: int = 256
    smem_runtime_overhead_per_block: int = 1024


ARCHES = {
    "a100": Arch(
        "A100 SM80",
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        max_warps_per_sm=64,
        regs_per_sm=65536,
        smem_per_sm=164 * 1024,
    ),
    "h100": Arch(
        "H100/GH200 SM90",
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        max_warps_per_sm=64,
        regs_per_sm=65536,
        smem_per_sm=228 * 1024,
    ),
}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def occupancy(arch: Arch, threads: int, regs_per_thread: int, smem_bytes: int) -> dict:
    if threads <= 0 or threads > 1024:
        raise ValueError("threads per block must be in 1..1024")
    warps_per_block = ceil_div(threads, 32)
    regs_per_warp = regs_per_thread * 32
    regs_per_warp = ceil_div(regs_per_warp, arch.reg_alloc_granularity_per_warp)
    regs_per_warp *= arch.reg_alloc_granularity_per_warp
    regs_per_block = regs_per_warp * warps_per_block
    smem_per_block = smem_bytes + arch.smem_runtime_overhead_per_block

    by_threads = arch.max_threads_per_sm // threads
    by_blocks = arch.max_blocks_per_sm
    by_warps = arch.max_warps_per_sm // warps_per_block
    by_regs = arch.regs_per_sm // regs_per_block if regs_per_block else by_blocks
    by_smem = arch.smem_per_sm // smem_per_block if smem_per_block else by_blocks
    active_blocks = min(by_threads, by_blocks, by_warps, by_regs, by_smem)
    active_warps = active_blocks * warps_per_block
    return {
        "active_blocks": active_blocks,
        "active_warps": active_warps,
        "occupancy": active_warps / arch.max_warps_per_sm,
        "limit_threads": by_threads,
        "limit_blocks": by_blocks,
        "limit_warps": by_warps,
        "limit_regs": by_regs,
        "limit_smem": by_smem,
        "regs_per_block": regs_per_block,
        "smem_per_block": smem_per_block,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, default="h100")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--regs", type=int, default=64, help="registers per thread")
    parser.add_argument("--smem-kib", type=float, default=32.0)
    args = parser.parse_args()

    arch = ARCHES[args.arch]
    result = occupancy(arch, args.threads, args.regs, int(args.smem_kib * 1024))
    print(f"Architecture: {arch.name}")
    print(f"Threads/block: {args.threads}")
    print(f"Registers/thread: {args.regs}")
    print(f"Dynamic+static shared memory requested: {args.smem_kib:.1f} KiB")
    print(f"Registers/block after warp granularity: {result['regs_per_block']}")
    print(f"Shared memory/block incl. CUDA overhead: {result['smem_per_block'] / 1024:.1f} KiB")
    print(f"Active blocks/SM: {result['active_blocks']}")
    print(f"Active warps/SM: {result['active_warps']} / {arch.max_warps_per_sm}")
    print(f"Theoretical occupancy: {100.0 * result['occupancy']:.1f}%")
    print("Block ceilings:")
    print(f"  threads: {result['limit_threads']}")
    print(f"  block limit: {result['limit_blocks']}")
    print(f"  warps: {result['limit_warps']}")
    print(f"  registers: {result['limit_regs']}")
    print(f"  shared memory: {result['limit_smem']}")


if __name__ == "__main__":
    main()
