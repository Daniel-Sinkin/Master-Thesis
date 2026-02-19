#!/usr/bin/env python3
"""Generate arithmetic-intensity heatmaps for rectangular GEMM-like kernels.

Run from repository root:
    python3 scripts/generate_arithmetic_intensity_heatmap.py
"""

from __future__ import annotations

import argparse
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


def pad_to(x: int, tile: int) -> int:
    return ((x + tile - 1) // tile) * tile


def ai_ideal(m: int, n: int, k: int, elem_bytes: int) -> float:
    # FP operations for GEMM-like contraction: C_{M x N} += A_{M x K} B_{K x N}
    flops = 2.0 * m * n * k
    # Traffic model assumes read A, read B, read+write C (beta != 0).
    bytes_moved = elem_bytes * (m * k + k * n + 2 * m * n)
    return flops / bytes_moved


def ai_effective_with_padding(
    m: int, n: int, k: int, tile: int, elem_bytes: int
) -> float:
    mp = pad_to(m, tile)
    np_ = pad_to(n, tile)
    kp = pad_to(k, tile)
    useful_flops = 2.0 * m * n * k
    padded_bytes = elem_bytes * (mp * kp + kp * np_ + 2 * mp * np_)
    return useful_flops / padded_bytes


def build_grid(min_dim: int, max_dim: int, step: int) -> tuple[list[int], list[int]]:
    dims = list(range(min_dim, max_dim + 1, step))
    return dims, dims


def render_heatmap(
    ms: list[int],
    ns: list[int],
    k: int,
    tile: int,
    elem_bytes: int,
    precision_label: str,
    output: Path,
) -> None:
    ideal = [[ai_ideal(m, n, k, elem_bytes) for n in ns] for m in ms]
    padded = [
        [ai_effective_with_padding(m, n, k, tile, elem_bytes) for n in ns] for m in ms
    ]
    vmin = min(min(row) for row in padded)
    vmax = max(max(row) for row in ideal)

    plt.rcParams.update(
        {
            "figure.figsize": (14.6, 6.2),
            "figure.dpi": 140,
            "font.size": 11.5,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    extent = [ns[0], ns[-1], ms[0], ms[-1]]
    cmap = "magma"

    im0 = axes[0].imshow(
        ideal,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    im1 = axes[1].imshow(
        padded,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    for ax in axes:
        ax.set_xlabel("N dimension")
        ax.set_ylabel("M dimension")

    axes[0].set_title(f"Ideal AI, K={k} (no padding)")
    axes[1].set_title(f"Effective AI, K={k}, tile={tile} (with padding)")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.93, pad=0.02)
    cbar.set_label("Arithmetic intensity [FLOP/byte]")

    fig.suptitle(
        (
            "Arithmetic-Intensity Landscape for Rectangular GEMM-Like Kernels "
            f"({precision_label})"
        ),
        y=1.03,
        fontsize=16,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=64, help="Fixed K dimension.")
    parser.add_argument("--tile", type=int, default=16, help="Padding tile size.")
    parser.add_argument("--min-dim", type=int, default=16)
    parser.add_argument("--max-dim", type=int, default=256)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--output-fp32",
        type=Path,
        default=Path("images/arithmetic_intensity_heatmap.pdf"),
        help="Output PDF path for FP32 heatmap.",
    )
    parser.add_argument(
        "--output-fp64",
        type=Path,
        default=Path("images/arithmetic_intensity_heatmap_fp64.pdf"),
        help="Output PDF path for FP64 heatmap.",
    )
    args = parser.parse_args()

    ms, ns = build_grid(args.min_dim, args.max_dim, args.step)

    render_heatmap(
        ms=ms,
        ns=ns,
        k=args.k,
        tile=args.tile,
        elem_bytes=4,
        precision_label="FP32",
        output=args.output_fp32,
    )
    render_heatmap(
        ms=ms,
        ns=ns,
        k=args.k,
        tile=args.tile,
        elem_bytes=8,
        precision_label="FP64",
        output=args.output_fp64,
    )


if __name__ == "__main__":
    main()
