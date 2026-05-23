#!/usr/bin/env python3
"""Generate first-pass PEPS CUDA benchmark cases.

The output is intentionally simple CSV so it can be pasted into a lab notebook,
loaded into pandas, or used as a Slurm array manifest later.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Gpu:
    key: str
    name: str
    hbm_gib: float


GPUS = {
    "a100_40": Gpu("a100_40", "A100 SXM 40GB", 40.0),
    "h100_80": Gpu("h100_80", "H100 SXM 80GB", 80.0),
    "h200_141": Gpu("h200_141", "H200 SXM 141GB", 141.0),
    "jupiter_gh200": Gpu("jupiter_gh200", "JUPITER GH200 96GB", 96.0),
}


def parse_int_list(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def gib(nbytes: float) -> float:
    return nbytes / (1024.0**3)


def classify_case(dense_o_gib: float, sampled_o_gib: float, hbm_gib: float) -> str:
    # Leave space for PEPS tensors, environments, cuBLAS/cuSOLVER workspace, and
    # profiler overhead. The labels are triage flags, not hard feasibility claims.
    budget = 0.70 * hbm_gib
    if dense_o_gib <= budget:
        return "dense-o-ok"
    if sampled_o_gib <= budget:
        return "sampled-sector-o-ok"
    return "direct-gram-required"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=GPUS, default="jupiter_gh200")
    parser.add_argument("--lattices", default="4x4,8x8,16x16,32x32")
    parser.add_argument("--d-values", default="2,4,6,8")
    parser.add_argument("--dc-values", default="16,32,64,96")
    parser.add_argument("--samples", default="128,512,2000,5000")
    parser.add_argument("--local-dim", type=int, default=2)
    parser.add_argument("--complex-bytes", type=int, default=16)
    args = parser.parse_args()

    gpu = GPUS[args.gpu]
    lattices = []
    for item in args.lattices.split(","):
        lx_text, ly_text = item.lower().split("x", maxsplit=1)
        lattices.append((int(lx_text), int(ly_text)))
    d_values = parse_int_list(args.d_values)
    dc_values = parse_int_list(args.dc_values)
    samples_values = parse_int_list(args.samples)

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "gpu",
            "lattice",
            "D",
            "Dc",
            "samples",
            "bulk_parameters",
            "sampled_sector_parameters",
            "dense_o_gib",
            "sampled_sector_o_gib",
            "gram_gib",
            "triage",
        ],
    )
    writer.writeheader()

    for (lx, ly), d, dc, samples in product(
        lattices, d_values, dc_values, samples_values
    ):
        sites = lx * ly
        bulk_parameters = sites * args.local_dim * (d**4)
        sampled_sector_parameters = sites * (d**4)
        dense_o_gib = gib(samples * bulk_parameters * args.complex_bytes)
        sampled_o_gib = gib(samples * sampled_sector_parameters * args.complex_bytes)
        gram_gib = gib(samples * samples * args.complex_bytes)
        writer.writerow(
            {
                "gpu": gpu.key,
                "lattice": f"{lx}x{ly}",
                "D": d,
                "Dc": dc,
                "samples": samples,
                "bulk_parameters": bulk_parameters,
                "sampled_sector_parameters": sampled_sector_parameters,
                "dense_o_gib": f"{dense_o_gib:.3f}",
                "sampled_sector_o_gib": f"{sampled_o_gib:.3f}",
                "gram_gib": f"{gram_gib:.3f}",
                "triage": classify_case(dense_o_gib, sampled_o_gib, gpu.hbm_gib),
            }
        )


if __name__ == "__main__":
    main()
