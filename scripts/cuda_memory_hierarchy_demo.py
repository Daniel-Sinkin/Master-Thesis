#!/usr/bin/env python3
"""Interactive CUDA memory hierarchy + cache miss visualizer.

This script shows two linked views:
1) How traffic splits across L1, L2, and DRAM.
2) A "cache miss rain" view across warps and load instructions.

Examples:
    python3 scripts/cuda_memory_hierarchy_demo.py
    python3 scripts/cuda_memory_hierarchy_demo.py --no-gui \
        --output images/cuda_memory_hierarchy_demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def effective_hit_rates(
    *,
    base_l1_hit: float,
    base_l2_hit_on_l1_miss: float,
    working_set_mb: float,
) -> tuple[float, float]:
    # Larger working sets tend to reduce cache locality in this simple model.
    pressure = clamp((working_set_mb - 0.5) / 11.5, 0.0, 1.0)
    l1_eff = clamp(base_l1_hit * (1.0 - 0.62 * pressure), 0.01, 0.99)
    l2_eff = clamp(base_l2_hit_on_l1_miss * (1.0 - 0.33 * pressure), 0.01, 0.99)
    return l1_eff, l2_eff


def hierarchy_split(
    *,
    request_count: int,
    l1_hit: float,
    l2_hit_on_l1_miss: float,
) -> tuple[int, int, int]:
    l1_hits = int(round(request_count * l1_hit))
    l1_misses = max(0, request_count - l1_hits)
    l2_hits = int(round(l1_misses * l2_hit_on_l1_miss))
    l2_hits = min(l2_hits, l1_misses)
    dram_misses = max(0, request_count - l1_hits - l2_hits)
    return l1_hits, l2_hits, dram_misses


def generate_miss_rain(
    np,
    *,
    warps: int,
    ops_per_warp: int,
    l1_hit: float,
    l2_hit_on_l1_miss: float,
    locality: float,
    seed: int,
):
    probs = np.array(
        [
            l1_hit,
            (1.0 - l1_hit) * l2_hit_on_l1_miss,
            (1.0 - l1_hit) * (1.0 - l2_hit_on_l1_miss),
        ],
        dtype=float,
    )
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed)
    levels = np.empty((warps, ops_per_warp), dtype=np.int8)

    for warp in range(warps):
        levels[warp, 0] = int(rng.choice(3, p=probs))
        for op in range(1, ops_per_warp):
            if rng.random() < locality:
                levels[warp, op] = levels[warp, op - 1]
            else:
                levels[warp, op] = int(rng.choice(3, p=probs))

    x = np.tile(np.arange(ops_per_warp, dtype=float), warps)
    y = np.repeat(np.arange(warps, dtype=float), ops_per_warp)
    x += rng.uniform(-0.36, 0.36, size=x.shape[0])
    y += rng.uniform(-0.34, 0.34, size=y.shape[0])
    return x, y, levels.ravel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=4096, help="Total memory requests.")
    parser.add_argument("--warps", type=int, default=24, help="Warps in the synthetic kernel.")
    parser.add_argument(
        "--ops-per-warp",
        type=int,
        default=128,
        help="Load instructions per warp.",
    )
    parser.add_argument("--l1-hit", type=float, default=0.72, help="Base L1 hit rate.")
    parser.add_argument(
        "--l2-hit",
        type=float,
        default=0.64,
        help="Base L2 hit rate on L1 misses.",
    )
    parser.add_argument("--locality", type=float, default=0.55, help="Temporal locality level.")
    parser.add_argument(
        "--working-set-mb",
        type=float,
        default=2.0,
        help="Working set size in MiB.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path for a snapshot.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Render one snapshot without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import matplotlib

        if args.no_gui:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib and numpy. Install with "
            "`python3 -m pip install matplotlib numpy`."
        ) from exc

    colors = {
        "L1 hit": "#22C55E",
        "L2 hit": "#F59E0B",
        "DRAM miss": "#EF4444",
    }
    level_palette = np.array([colors["L1 hit"], colors["L2 hit"], colors["DRAM miss"]], dtype=object)
    latency_cycles = {"L1 hit": 20.0, "L2 hit": 125.0, "DRAM miss": 520.0}

    def render_view(
        *,
        ax_flow,
        ax_rain,
        request_count: int,
        warps: int,
        ops_per_warp: int,
        base_l1_hit: float,
        base_l2_hit: float,
        locality: float,
        working_set_mb: float,
    ) -> None:
        l1_eff, l2_eff = effective_hit_rates(
            base_l1_hit=base_l1_hit,
            base_l2_hit_on_l1_miss=base_l2_hit,
            working_set_mb=working_set_mb,
        )
        l1_hits, l2_hits, dram_misses = hierarchy_split(
            request_count=request_count,
            l1_hit=l1_eff,
            l2_hit_on_l1_miss=l2_eff,
        )
        avg_latency = (
            l1_hits * latency_cycles["L1 hit"]
            + l2_hits * latency_cycles["L2 hit"]
            + dram_misses * latency_cycles["DRAM miss"]
        ) / max(1, request_count)

        ax_flow.clear()
        ax_flow.set_xlim(0, request_count)
        ax_flow.set_ylim(-0.65, 0.65)
        ax_flow.set_yticks([])
        ax_flow.set_xlabel("Memory Requests")
        ax_flow.grid(axis="x", linestyle="--", alpha=0.28)

        parts = [("L1 hit", l1_hits), ("L2 hit", l2_hits), ("DRAM miss", dram_misses)]
        left = 0
        handles = []
        labels = []
        for label, value in parts:
            bar = ax_flow.barh(
                [0],
                [value],
                left=[left],
                color=colors[label],
                edgecolor="#111827",
                linewidth=0.6,
                height=0.42,
            )
            handles.append(bar[0])
            labels.append(label)
            if value > 0:
                share = 100.0 * value / max(1, request_count)
                text = f"{label}: {share:.1f}%"
                text_x = left + (value / 2.0)
                if value >= request_count * 0.08:
                    ax_flow.text(
                        text_x,
                        0,
                        text,
                        ha="center",
                        va="center",
                        color="#0B1020",
                        fontsize=9.6,
                        weight="bold",
                    )
                else:
                    ax_flow.text(
                        text_x,
                        0.36,
                        text,
                        ha="center",
                        va="center",
                        color="#111827",
                        fontsize=9.0,
                    )
            left += value

        miss_rate = dram_misses / max(1, request_count)
        ax_flow.set_title(
            "CUDA Memory Hierarchy Traffic\n"
            f"effective L1={l1_eff:.2f}, effective L2={l2_eff:.2f}, "
            f"DRAM miss rate={miss_rate:.2%}, avg latency={avg_latency:.1f} cycles"
        )
        ax_flow.legend(handles, labels, loc="upper right")

        seed = (
            int(base_l1_hit * 1_000) * 7
            + int(base_l2_hit * 1_000) * 13
            + int(locality * 1_000) * 17
            + int(working_set_mb * 100) * 19
            + request_count
        )
        x, y, levels = generate_miss_rain(
            np,
            warps=warps,
            ops_per_warp=ops_per_warp,
            l1_hit=l1_eff,
            l2_hit_on_l1_miss=l2_eff,
            locality=locality,
            seed=seed,
        )

        ax_rain.clear()
        ax_rain.scatter(x, y, c=level_palette[levels], s=12.0, linewidths=0.0, alpha=0.82)
        ax_rain.set_xlim(-1.0, float(ops_per_warp))
        ax_rain.set_ylim(-1.0, float(warps))
        ax_rain.set_xlabel("Load Instruction Index")
        ax_rain.set_ylabel("Warp ID")
        ax_rain.set_title("Cache Miss Rain (green=L1 hit, amber=L2 hit, red=DRAM miss)")
        ax_rain.grid(alpha=0.20, linestyle=":")

    if args.no_gui:
        fig, (ax_flow, ax_rain) = plt.subplots(
            2,
            1,
            figsize=(13.0, 8.4),
            gridspec_kw={"height_ratios": [1.0, 2.1]},
        )
        render_view(
            ax_flow=ax_flow,
            ax_rain=ax_rain,
            request_count=max(64, args.requests),
            warps=max(1, args.warps),
            ops_per_warp=max(8, args.ops_per_warp),
            base_l1_hit=clamp(args.l1_hit, 0.01, 0.99),
            base_l2_hit=clamp(args.l2_hit, 0.01, 0.99),
            locality=clamp(args.locality, 0.0, 0.98),
            working_set_mb=max(0.05, args.working_set_mb),
        )
        fig.tight_layout()
        output_path = args.output or Path("images/cuda_memory_hierarchy_demo.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=165, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated {output_path}")
        return

    from matplotlib.widgets import Slider

    plt.rcParams.update(
        {
            "figure.figsize": (13.0, 8.4),
            "font.size": 10.5,
            "axes.titlesize": 12.8,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
        }
    )
    fig, (ax_flow, ax_rain) = plt.subplots(
        2,
        1,
        gridspec_kw={"height_ratios": [1.0, 2.1]},
    )
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.30, hspace=0.35)

    slider_l1_ax = fig.add_axes((0.12, 0.20, 0.74, 0.03))
    slider_l2_ax = fig.add_axes((0.12, 0.15, 0.74, 0.03))
    slider_locality_ax = fig.add_axes((0.12, 0.10, 0.74, 0.03))
    slider_ws_ax = fig.add_axes((0.12, 0.05, 0.74, 0.03))

    slider_l1 = Slider(
        slider_l1_ax,
        "Base L1 Hit",
        0.05,
        0.95,
        valinit=clamp(args.l1_hit, 0.05, 0.95),
        valstep=0.01,
    )
    slider_l2 = Slider(
        slider_l2_ax,
        "Base L2 Hit",
        0.05,
        0.95,
        valinit=clamp(args.l2_hit, 0.05, 0.95),
        valstep=0.01,
    )
    slider_locality = Slider(
        slider_locality_ax,
        "Locality",
        0.00,
        0.95,
        valinit=clamp(args.locality, 0.0, 0.95),
        valstep=0.01,
    )
    slider_working_set = Slider(
        slider_ws_ax,
        "Working Set [MiB]",
        0.25,
        12.0,
        valinit=clamp(args.working_set_mb, 0.25, 12.0),
        valstep=0.05,
    )

    def redraw(_) -> None:
        render_view(
            ax_flow=ax_flow,
            ax_rain=ax_rain,
            request_count=max(64, args.requests),
            warps=max(1, args.warps),
            ops_per_warp=max(8, args.ops_per_warp),
            base_l1_hit=slider_l1.val,
            base_l2_hit=slider_l2.val,
            locality=slider_locality.val,
            working_set_mb=slider_working_set.val,
        )
        fig.canvas.draw_idle()

    slider_l1.on_changed(redraw)
    slider_l2.on_changed(redraw)
    slider_locality.on_changed(redraw)
    slider_working_set.on_changed(redraw)

    redraw(None)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=165, bbox_inches="tight")
        print(f"Generated {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
