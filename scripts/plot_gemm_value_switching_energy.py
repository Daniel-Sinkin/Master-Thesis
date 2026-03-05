#!/usr/bin/env python3
"""Plot GEMM value-pattern power and normalized energy from power logs.

Run from repository root:
    python3 scripts/plot_gemm_value_switching_energy.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with `python3 -m pip install matplotlib`."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "code" / "profiling" / "phenomena" / "results"
DEFAULT_OUTPUT = ROOT / "images" / "gemm_value_switching_energy.pdf"

TIME_FMT = "%Y/%m/%d %H:%M:%S.%f"

MODE_ORDER = [
    "all_zero",
    "zero_every_2",
    "zero_every_3",
    "zero_every_4",
    "zero_every_5",
    "normal",
    "uniform",
]


@dataclass
class ModeSummary:
    mode: str
    active_gpu: int
    mean_power_w: float
    median_power_w: float
    active_seconds: float
    norm_energy_120s_j: float


def pretty_mode(mode: str) -> str:
    mapping = {
        "all_zero": "All Zero",
        "zero_every_2": "Zero Every 2nd",
        "zero_every_3": "Zero Every 3rd",
        "zero_every_4": "Zero Every 4th",
        "zero_every_5": "Zero Every 5th",
        "normal": "Normal",
        "uniform": "Uniform",
    }
    return mapping.get(mode, mode.replace("_", " ").title())


def median(values: list[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def parse_mode_file(path: Path) -> ModeSummary | None:
    by_gpu: dict[int, dict[str, list[float] | list[datetime]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
                power = float(row["power_draw_w"])
                util = float(row["utilization_gpu_pct"])
                ts = datetime.strptime(row["timestamp"], TIME_FMT)
            except Exception:
                continue
            if idx not in by_gpu:
                by_gpu[idx] = {"power": [], "util": [], "time": []}
            by_gpu[idx]["power"].append(power)
            by_gpu[idx]["util"].append(util)
            by_gpu[idx]["time"].append(ts)

    if not by_gpu:
        return None

    # Identify the active GPU by highest mean utilization in this file.
    active_gpu = max(
        by_gpu.keys(),
        key=lambda g: (
            sum(by_gpu[g]["util"]) / max(len(by_gpu[g]["util"]), 1),  # type: ignore[arg-type]
            len(by_gpu[g]["util"]),  # type: ignore[arg-type]
        ),
    )

    power = by_gpu[active_gpu]["power"]  # type: ignore[assignment]
    util = by_gpu[active_gpu]["util"]  # type: ignore[assignment]
    time = by_gpu[active_gpu]["time"]  # type: ignore[assignment]
    if not power or not time:
        return None

    active_samples = [
        (t, p)
        for t, p, u in zip(time, power, util)
        if u > 0.0
    ]
    if len(active_samples) < 2:
        return None

    active_samples.sort(key=lambda x: x[0])
    active_p = [p for _, p in active_samples]
    mean_power = sum(active_p) / len(active_p)
    median_power = median(active_p)
    active_seconds = (active_samples[-1][0] - active_samples[0][0]).total_seconds()
    norm_energy_120s = mean_power * 120.0

    mode = path.stem.replace("09_gemm_value_switching_power_", "")
    return ModeSummary(
        mode=mode,
        active_gpu=active_gpu,
        mean_power_w=mean_power,
        median_power_w=median_power,
        active_seconds=active_seconds,
        norm_energy_120s_j=norm_energy_120s,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("09_gemm_value_switching_power_*.csv"))
    summaries: list[ModeSummary] = []
    for f in files:
        s = parse_mode_file(f)
        if s is not None:
            summaries.append(s)

    if not summaries:
        raise SystemExit("No valid 09_gemm_value_switching power CSV files found.")

    order_map = {mode: i for i, mode in enumerate(MODE_ORDER)}
    summaries.sort(key=lambda s: (order_map.get(s.mode, 999), s.mode))

    labels = [pretty_mode(s.mode) for s in summaries]
    mean_power = [s.mean_power_w for s in summaries]
    norm_energy = [s.norm_energy_120s_j / 1000.0 for s in summaries]  # kJ
    x = list(range(len(summaries)))

    plt.rcParams.update(
        {
            "figure.figsize": (12.4, 5.8),
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )

    fig, (ax0, ax1) = plt.subplots(1, 2)

    ax0.bar(x, mean_power, color="#1f77b4", width=0.64)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=15, ha="right")
    ax0.set_ylabel("Mean active GPU power [W]")
    ax0.set_title("GEMM Value-Pattern Power")

    ax1.bar(x, norm_energy, color="#ff7f0e", width=0.64)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Normalized energy for 120 s [kJ]")
    ax1.set_title("GEMM Value-Pattern Energy")

    fig.suptitle("A100 SGEMM Switching-Pattern Comparison (active GPU only)", y=1.02)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote plot: {args.output}")
    print("Mode summary:")
    for s in summaries:
        print(
            f"  {s.mode}: gpu={s.active_gpu} mean_power={s.mean_power_w:.2f} W "
            f"median_power={s.median_power_w:.2f} W active_s={s.active_seconds:.3f} "
            f"normE120={s.norm_energy_120s_j:.1f} J"
        )


if __name__ == "__main__":
    main()

