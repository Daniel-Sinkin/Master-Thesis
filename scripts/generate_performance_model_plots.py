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
    device_label: str
    bandwidth_tbps: float
    fp64_tflops: float
    fp32_tflops: float
    line_style: str
    marker_style: str


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


def _plot_roofline_on_axis(
    ax: plt.Axes,
    cfg: RooflineConfig,
    xmax: float = 25.0,
    *,
    include_precision_legend: bool,
) -> None:
    x_points = [xmax * i / 1400.0 for i in range(1401)]

    y_fp64 = [roofline_value(x, cfg.fp64_tflops, cfg.bandwidth_tbps) for x in x_points]
    y_fp32 = [roofline_value(x, cfg.fp32_tflops, cfg.bandwidth_tbps) for x in x_points]

    i64 = cfg.fp64_tflops / cfg.bandwidth_tbps
    i32 = cfg.fp32_tflops / cfg.bandwidth_tbps

    fp64_label = "FP64" if include_precision_legend else None
    fp32_label = "FP32" if include_precision_legend else None
    ax.plot(
        x_points,
        y_fp64,
        color="#D62728",
        linestyle=cfg.line_style,
        linewidth=2.8,
        label=fp64_label,
    )
    ax.plot(
        x_points,
        y_fp32,
        color="#FF7F0E",
        linestyle=cfg.line_style,
        linewidth=2.8,
        label=fp32_label,
    )

    guides = [
        (i64, cfg.fp64_tflops, "#D62728", "FP64"),
        (i32, cfg.fp32_tflops, "#FF7F0E", "FP32"),
    ]
    for ix, ypeak, color, label in guides:
        ax.scatter(
            [ix],
            [ypeak],
            color=color,
            s=34,
            marker=cfg.marker_style,
            zorder=5,
        )
        if label == "FP64":
            # Anchor text box at top-left so it sits to the right and below the ridge point.
            ax.annotate(
                f"{cfg.device_label} I*_{label}={ix:.1f}",
                xy=(ix, ypeak),
                xytext=(ix + 0.45, ypeak - 1.4),
                textcoords="data",
                ha="left",
                va="top",
                fontsize=10.5,
                color=color,
                weight="bold",
            )
        else:
            ax.annotate(
                f"{cfg.device_label} I*_{label}={ix:.1f}",
                xy=(ix, ypeak),
                xytext=(ix + 0.45, ypeak + 1.4),
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=10.5,
                color=color,
                weight="bold",
            )

    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("Operational intensity I [FLOP/byte]")


def generate_combined_roofline_plot(configs: list[RooflineConfig], output_name: str, xmax: float = 25.0) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.9))
    ymax = max(cfg.fp32_tflops for cfg in configs) * 1.08
    for idx, cfg in enumerate(configs):
        _plot_roofline_on_axis(
            ax,
            cfg,
            xmax=xmax,
            include_precision_legend=(idx == 0),
        )

    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("Performance [TFLOP/s]")
    ax.set_title("Roofline Model - NVIDIA A100-SXM4 40GB vs NVIDIA H100-SXM5 80GB")
    ax.legend(loc="upper left")

    # Device style note without adding extra legend entries.
    ax.text(
        0.995,
        0.02,
        "A100: solid, marker o    H100: dashed, marker ^",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#444444",
    )

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
            device_label="A100-40GB",
            bandwidth_tbps=1.555,
            fp64_tflops=9.7,
            fp32_tflops=19.5,
            line_style="-",
            marker_style="o",
        ),
        RooflineConfig(
            device_label="H100",
            bandwidth_tbps=3.350,
            fp64_tflops=33.5,
            fp32_tflops=67.0,
            line_style="--",
            marker_style="^",
        ),
    ]

    generate_combined_roofline_plot(
        roofline_configs,
        output_name="roofline_a100_h100_combined.pdf",
        xmax=25.0,
    )
    generate_amdahl_plot()

    print("Generated:")
    print("  images/roofline_a100_h100_combined.pdf")
    print("  images/amdahl_speedup.pdf")


if __name__ == "__main__":
    main()
