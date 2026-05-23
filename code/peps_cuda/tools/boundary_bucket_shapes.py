#!/usr/bin/env python3
"""Emit approximate GEMM shape buckets for PEPS boundary row absorption.

The formulas follow `research/peps_cuda/boundary_mps_lowering.md`. They are not
a profiler replacement; they are a manifest generator for first grouped-GEMM,
cuBLASLt, and cuTENSOR microbenchmarks.
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmShape:
    family: str
    m: int
    n: int
    k: int


def leg_dims(x: int, y: int, lx: int, ly: int, bond_dim: int) -> tuple[int, int, int, int]:
    north = 1 if y == 0 else bond_dim
    east = 1 if x == lx - 1 else bond_dim
    south = 1 if y == ly - 1 else bond_dim
    west = 1 if x == 0 else bond_dim
    return north, east, south, west


def boundary_chi(x: int, lx: int, chi: int) -> tuple[int, int]:
    left = 1 if x == 0 else chi
    right = 1 if x == lx - 1 else chi
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx", type=int, default=16)
    parser.add_argument("--ly", type=int, default=16)
    parser.add_argument("--d", type=int, default=8, help="PEPS bond dimension")
    parser.add_argument("--dc", type=int, default=64, help="single-layer boundary chi")
    parser.add_argument(
        "--dc-double",
        type=int,
        default=None,
        help="double-layer boundary chi; defaults to --dc",
    )
    parser.add_argument(
        "--families",
        default="single,double",
        help="comma-separated subset of: single,double",
    )
    args = parser.parse_args()

    if args.lx <= 0 or args.ly <= 0 or args.d <= 0 or args.dc <= 0:
        raise SystemExit("lx, ly, d, and dc must be positive")
    dc_double = args.dc if args.dc_double is None else args.dc_double
    families = {item.strip() for item in args.families.split(",") if item.strip()}

    counts: Counter[GemmShape] = Counter()
    for y in range(args.ly):
        for x in range(args.lx):
            north, east, south, west = leg_dims(x, y, args.lx, args.ly, args.d)
            if "single" in families:
                chi_l, chi_r = boundary_chi(x, args.lx, args.dc)
                counts[
                    GemmShape(
                        "single_absorb",
                        chi_l * chi_r,
                        west * east * south,
                        north,
                    )
                ] += 1
            if "double" in families:
                chi_l, chi_r = boundary_chi(x, args.lx, dc_double)
                counts[
                    GemmShape(
                        "double_absorb",
                        chi_l * chi_r,
                        (west * east * south) ** 2,
                        north**2,
                    )
                ] += 1

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "family",
            "M",
            "N",
            "K",
            "count",
            "complex_fma_count",
            "real_flop_equiv_8mnk",
        ],
    )
    writer.writeheader()
    for shape, count in sorted(counts.items(), key=lambda item: (item[0].family, item[0].m, item[0].n, item[0].k)):
        mnk = shape.m * shape.n * shape.k
        writer.writerow(
            {
                "family": shape.family,
                "M": shape.m,
                "N": shape.n,
                "K": shape.k,
                "count": count,
                "complex_fma_count": mnk * count,
                "real_flop_equiv_8mnk": 8 * mnk * count,
            }
        )


if __name__ == "__main__":
    main()
