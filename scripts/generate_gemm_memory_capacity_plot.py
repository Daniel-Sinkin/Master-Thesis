#!/usr/bin/env python3
"""Generate GEMM storage-footprint plots against A100 capacity limits.

Model: square GEMM with dimensions (k, k, k) and storage for A, B, C only.
Memory footprint:
    M(k, s) = 3 * k^2 * s
where s is bytes per element.

Run from repository root:
    python3 scripts/generate_gemm_memory_capacity_plot.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with "
        "`python3 -m pip install matplotlib`."
    ) from exc


def gemm_storage_bytes(k: float, elem_bytes: int) -> float:
    return 3.0 * (k**2) * elem_bytes


def k_threshold(capacity_bytes: float, elem_bytes: int) -> float:
    return math.sqrt(capacity_bytes / (3.0 * elem_bytes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hbm-gb",
        type=float,
        default=40.0,
        help="A100 HBM capacity in GB (decimal).",
    )
    parser.add_argument(
        "--shared-kib",
        type=float,
        default=164.0,
        help="A100 shared memory capacity per SM in KiB.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images/gemm_memory_capacity_a100.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    hbm_bytes = args.hbm_gb * 1e9
    shared_bytes = args.shared_kib * 1024.0

    k_fp32_hbm = k_threshold(hbm_bytes, 4)
    k_fp64_hbm = k_threshold(hbm_bytes, 8)
    k_fp32_shared = k_threshold(shared_bytes, 4)
    k_fp64_shared = k_threshold(shared_bytes, 8)

    kmax_hbm = int(math.ceil(k_fp32_hbm * 1.15))
    k_hbm = list(range(1, kmax_hbm + 1))
    fp32_hbm_curve = [gemm_storage_bytes(k, 4) / 1e9 for k in k_hbm]
    fp64_hbm_curve = [gemm_storage_bytes(k, 8) / 1e9 for k in k_hbm]

    kmax_shared = max(256, int(math.ceil(k_fp32_shared * 2.0)))
    k_shared = list(range(1, kmax_shared + 1))
    fp32_shared_curve = [gemm_storage_bytes(k, 4) / 1024.0 for k in k_shared]
    fp64_shared_curve = [gemm_storage_bytes(k, 8) / 1024.0 for k in k_shared]

    plt.rcParams.update(
        {
            "figure.figsize": (14.6, 6.2),
            "figure.dpi": 140,
            "font.size": 11.5,
            "axes.titlesize": 14.5,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )

    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    # Panel 1: HBM capacity.
    ax0 = axes[0]
    ax0.plot(k_hbm, fp32_hbm_curve, color="#1F77B4", linewidth=2.8, label="FP32 storage")
    ax0.plot(k_hbm, fp64_hbm_curve, color="#D62728", linewidth=2.8, label="FP64 storage")
    ax0.axhline(args.hbm_gb, color="#444444", linewidth=1.7, linestyle=(0, (4, 3)), label=f"A100 HBM ({args.hbm_gb:.0f} GB)")
    ax0.axvline(k_fp32_hbm, color="#1F77B4", linewidth=1.4, linestyle=(0, (2, 3)))
    ax0.axvline(k_fp64_hbm, color="#D62728", linewidth=1.4, linestyle=(0, (2, 3)))
    ax0.text(k_fp32_hbm + 800, args.hbm_gb + 2.2, f"k≈{k_fp32_hbm:.0f} (FP32)", color="#1F77B4")
    ax0.text(k_fp64_hbm + 800, args.hbm_gb + 7.0, f"k≈{k_fp64_hbm:.0f} (FP64)", color="#D62728")
    ax0.set_xlim(0, kmax_hbm)
    ax0.set_ylim(0, max(fp64_hbm_curve) * 1.05)
    ax0.set_xlabel("Square dimension k for GEMM (k, k, k)")
    ax0.set_ylabel("Storage for A+B+C [GB]")
    ax0.set_title("Global-Memory Footprint vs A100 HBM")
    ax0.grid(True)
    ax0.legend(loc="upper left")

    # Panel 2: shared-memory limit (same model, zoomed for small k).
    ax1 = axes[1]
    ax1.plot(k_shared, fp32_shared_curve, color="#1F77B4", linewidth=2.8, label="FP32 storage")
    ax1.plot(k_shared, fp64_shared_curve, color="#D62728", linewidth=2.8, label="FP64 storage")
    ax1.axhline(args.shared_kib, color="#444444", linewidth=1.7, linestyle=(0, (4, 3)), label=f"A100 shared memory ({args.shared_kib:.0f} KiB)")
    ax1.axvline(k_fp32_shared, color="#1F77B4", linewidth=1.4, linestyle=(0, (2, 3)))
    ax1.axvline(k_fp64_shared, color="#D62728", linewidth=1.4, linestyle=(0, (2, 3)))
    ax1.text(k_fp32_shared + 6, args.shared_kib + 65, f"k≈{k_fp32_shared:.0f} (FP32)", color="#1F77B4")
    ax1.text(k_fp64_shared + 6, args.shared_kib + 140, f"k≈{k_fp64_shared:.0f} (FP64)", color="#D62728")
    ax1.set_xlim(0, kmax_shared)
    ax1.set_ylim(0, max(fp64_shared_curve) * 1.02)
    ax1.set_xlabel("Square dimension k for GEMM (k, k, k)")
    ax1.set_ylabel("Storage for A+B+C [KiB]")
    ax1.set_title("Same Footprint Model vs Per-SM Shared-Memory Limit")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    fig.suptitle(
        "GEMM Storage Capacity Model for A100 (A, B, C only; no workspace)",
        y=1.03,
        fontsize=16,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Generated {args.output}")
    print(
        "HBM thresholds: "
        f"FP32 k≈{k_fp32_hbm:.0f}, FP64 k≈{k_fp64_hbm:.0f}; "
        "Shared thresholds: "
        f"FP32 k≈{k_fp32_shared:.0f}, FP64 k≈{k_fp64_shared:.0f}"
    )


if __name__ == "__main__":
    main()
