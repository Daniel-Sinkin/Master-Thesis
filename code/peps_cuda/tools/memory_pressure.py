#!/usr/bin/env python3
"""Estimate PEPS working-set sizes for precision and layout arguments.

The numbers are intentionally explicit and a bit conservative: they are meant to
decide whether a benchmark belongs in dense-O debug mode, compact sampled-sector
mode, or direct Gram accumulation.
"""

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Precision:
    name: str
    complex_bytes: int
    real_bytes: int


PRECISIONS = [
    Precision("complex_fp64", 16, 8),
    Precision("complex_fp32", 8, 4),
    Precision("complex_fp16_storage", 4, 2),
]


def fmt_bytes(nbytes: float) -> str:
    value = float(nbytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def padded_site_parameter_count(lx: int, ly: int, local_dim: int, d: int) -> int:
    return lx * ly * local_dim * d**4


def open_boundary_parameter_count(lx: int, ly: int, local_dim: int, d: int) -> int:
    total = 0
    for x in range(lx):
        for y in range(ly):
            north = 1 if x == 0 else d
            south = 1 if x == lx - 1 else d
            west = 1 if y == 0 else d
            east = 1 if y == ly - 1 else d
            total += local_dim * north * east * south * west
    return total


def boundary_layer_elements(lx: int, ly: int, dc: int, d: int) -> int:
    # Row-boundary storage approximation for single-layer MPS tensors over a row.
    # This ignores ragged edge effects but is the right order for buffer planning.
    return max(lx, ly) * dc * dc * d


def double_boundary_layer_elements(lx: int, ly: int, dc_double: int, d: int) -> int:
    return max(lx, ly) * dc_double * dc_double * d * d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lx", type=int, default=16)
    parser.add_argument("--ly", type=int, default=16)
    parser.add_argument("--d", type=int, default=8, help="PEPS bond dimension")
    parser.add_argument("--dc", type=int, default=64, help="single-layer boundary dimension")
    parser.add_argument(
        "--dc-double",
        type=int,
        default=16,
        help="double-layer sampling boundary dimension",
    )
    parser.add_argument("--local-dim", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--hbm-gb", type=float, default=96.0)
    args = parser.parse_args()

    padded_params = padded_site_parameter_count(
        args.lx, args.ly, args.local_dim, args.d
    )
    open_params = open_boundary_parameter_count(
        args.lx, args.ly, args.local_dim, args.d
    )
    sampled_sector_params = padded_params // args.local_dim
    gram_elements = args.samples * args.samples
    single_env_elements = boundary_layer_elements(args.lx, args.ly, args.dc, args.d)
    double_env_elements = double_boundary_layer_elements(
        args.lx, args.ly, args.dc_double, args.d
    )
    hbm_bytes = args.hbm_gb * (1024**3)

    print(
        "scenario,"
        f"L={args.lx}x{args.ly},D={args.d},d={args.local_dim},"
        f"Dc={args.dc},Dc_double={args.dc_double},Ns={args.samples},"
        f"HBM={args.hbm_gb:.1f}GiB"
    )
    print(
        "precision,peps_open,peps_padded,dense_O,sampled_sector_O,"
        "sample_gram,single_env_pair,double_env_pair,dense_total,sampled_total,"
        "dense_hbm_pct,sampled_hbm_pct"
    )
    for precision in PRECISIONS:
        peps_open = open_params * precision.complex_bytes
        peps_padded = padded_params * precision.complex_bytes
        dense_o = args.samples * padded_params * precision.complex_bytes
        sampled_o = args.samples * sampled_sector_params * precision.complex_bytes
        gram = gram_elements * precision.complex_bytes
        single_env_pair = 2 * single_env_elements * precision.complex_bytes
        double_env_pair = 2 * double_env_elements * precision.complex_bytes
        sample_spins = args.samples * args.lx * args.ly * 4
        energies = args.samples * precision.complex_bytes
        weights = args.samples * precision.real_bytes
        dense_total = (
            peps_padded
            + dense_o
            + gram
            + single_env_pair
            + double_env_pair
            + sample_spins
            + energies
            + weights
        )
        sampled_total = (
            peps_padded
            + sampled_o
            + gram
            + single_env_pair
            + double_env_pair
            + sample_spins
            + energies
            + weights
        )
        print(
            f"{precision.name},"
            f"{fmt_bytes(peps_open)},"
            f"{fmt_bytes(peps_padded)},"
            f"{fmt_bytes(dense_o)},"
            f"{fmt_bytes(sampled_o)},"
            f"{fmt_bytes(gram)},"
            f"{fmt_bytes(single_env_pair)},"
            f"{fmt_bytes(double_env_pair)},"
            f"{fmt_bytes(dense_total)},"
            f"{fmt_bytes(sampled_total)},"
            f"{100.0 * dense_total / hbm_bytes:.1f},"
            f"{100.0 * sampled_total / hbm_bytes:.1f}"
        )


if __name__ == "__main__":
    main()
