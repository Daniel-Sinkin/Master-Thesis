#!/usr/bin/env python3
"""Generate analytical performance-model plots for Chapter 3.

Run from repository root:
    python3 scripts/generate_performance_model_plots.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install it with "
        "`python3 -m pip install matplotlib`."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"


@dataclass(frozen=True)
class RooflineConfig:
    title: str
    bandwidth_tbps: float
    fp64_tflops: float
    fp32_tflops: float
    ymax: float


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12.0, 6.8),
            "figure.dpi": 140,
            "font.size": 12,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.color": "#9AA0A6",
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#B8BEC4",
        }
    )


def roofline_value(intensity: float, peak_tflops: float, bandwidth_tbps: float) -> float:
    return min(peak_tflops, bandwidth_tbps * intensity)


def _plot_roofline_on_axis(ax: plt.Axes, cfg: RooflineConfig, xmax: float = 20.0) -> None:
    x_points = [xmax * i / 1400.0 for i in range(1401)]

    y_fp64 = [roofline_value(x, cfg.fp64_tflops, cfg.bandwidth_tbps) for x in x_points]
    y_fp32 = [roofline_value(x, cfg.fp32_tflops, cfg.bandwidth_tbps) for x in x_points]

    i64 = cfg.fp64_tflops / cfg.bandwidth_tbps
    i32 = cfg.fp32_tflops / cfg.bandwidth_tbps

    ax.plot(x_points, y_fp64, color="#D62728", linewidth=3.0, label=f"FP64 roof ({cfg.fp64_tflops:.1f})")
    ax.plot(x_points, y_fp32, color="#FF7F0E", linewidth=3.0, label=f"FP32 roof ({cfg.fp32_tflops:.1f})")

    guides = [
        (i64, cfg.fp64_tflops, "#D62728", "FP64"),
        (i32, cfg.fp32_tflops, "#FF7F0E", "FP32"),
    ]
    for ix, ypeak, color, label in guides:
        ax.axvline(ix, color="#8A8A8A", linestyle=(0, (2, 3)), linewidth=1.3, zorder=0)
        x_text = min(ix + 0.35, xmax - 3.0)
        ax.text(
            x_text,
            ypeak + 0.035 * cfg.ymax,
            f"I*_{label}={ix:.1f}",
            color=color,
            fontsize=12,
            weight="bold",
        )

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, cfg.ymax)
    ax.set_xlabel("Operational intensity I [FLOP/byte]")
    ax.set_ylabel("Performance [TFLOP/s]")
    ax.set_title(cfg.title)
    ax.legend(loc="upper left")


def generate_combined_roofline_plot(configs: list[RooflineConfig], output_name: str, xmax: float = 20.0) -> None:
    fig, axes = plt.subplots(1, len(configs), figsize=(15.6, 6.8), sharex=True)
    if len(configs) == 1:
        axes = [axes]
    for ax, cfg in zip(axes, configs):
        _plot_roofline_on_axis(ax, cfg, xmax=xmax)
    fig.tight_layout()
    fig.savefig(IMAGES / output_name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def amdahl_speedup(parallel_workers: float, parallel_fraction: float) -> float:
    return 1.0 / ((1.0 - parallel_fraction) + (parallel_fraction / parallel_workers))


def generate_amdahl_plot() -> None:
    max_exponent = int(math.log2(128))
    dense_p = [2 ** (i / 20.0) for i in range(0, max_exponent * 20 + 1)]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    curves = [
        (0.90, "#2C7FB8"),
        (0.95, "#FF7F0E"),
        (0.99, "#2CA02C"),
    ]
    for frac, color in curves:
        y = [amdahl_speedup(p, frac) for p in dense_p]
        ax.plot(dense_p, y, color=color, linewidth=3.1, label=f"f = {frac:.2f}")

    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 128)
    ax.set_ylim(1, 110)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("Parallel workers p")
    ax.set_ylabel("Speedup S(p)")
    ax.set_title("Amdahl Speedup Bound")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES / "amdahl_speedup.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    IMAGES.mkdir(exist_ok=True)

    roofline_configs = [
        RooflineConfig(
            title="Roofline Model - NVIDIA A100-SXM4 80GB",
            bandwidth_tbps=2.039,
            fp64_tflops=9.7,
            fp32_tflops=19.5,
            ymax=30.0,
        ),
        RooflineConfig(
            title="Roofline Model - NVIDIA H100-SXM5 80GB",
            bandwidth_tbps=3.350,
            fp64_tflops=33.5,
            fp32_tflops=67.0,
            ymax=80.0,
        ),
    ]

    generate_combined_roofline_plot(
        roofline_configs,
        output_name="roofline_a100_h100_combined.pdf",
        xmax=20.0,
    )
    generate_amdahl_plot()

    print("Generated:")
    print("  images/roofline_a100_h100_combined.pdf")
    print("  images/amdahl_speedup.pdf")


if __name__ == "__main__":
    main()
