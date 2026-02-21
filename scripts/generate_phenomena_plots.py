#!/usr/bin/env python3
"""Generate thesis-ready plots from profiling/phenomena CSV outputs.

Run from repository root:
    python3 scripts/generate_phenomena_plots.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install with "
        "`python3 -m pip install matplotlib`."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "code" / "profiling" / "phenomena" / "results"
DEFAULT_OUTPUT = ROOT / "images"


@dataclass(frozen=True)
class DatasetSpec:
    prefix: str
    output_pdf: str


SPECS = [
    DatasetSpec("01_divergence_sweep", "phenomena_divergence_sweep.pdf"),
    DatasetSpec("02_coalescing_sweep", "phenomena_coalescing_sweep.pdf"),
    DatasetSpec("03_shared_bank_conflict", "phenomena_shared_bank_conflict.pdf"),
    DatasetSpec("04_launch_overhead", "phenomena_launch_overhead.pdf"),
    DatasetSpec("05_ilp_dependency", "phenomena_ilp_dependency.pdf"),
    DatasetSpec("06_register_pressure", "phenomena_register_pressure.pdf"),
    DatasetSpec("07_arithmetic_intensity", "phenomena_arithmetic_intensity.pdf"),
]

# Inline fallback datasets from measured runs (used when no CSV exists yet).
INLINE_FALLBACK_ROWS: dict[str, list[dict[str, str]]] = {
    "01_divergence_sweep": [
        {"divergent_fraction": "0.0000", "avg_ms": "0.375788", "effective_tflops": "11.429245"},
        {"divergent_fraction": "0.1250", "avg_ms": "0.437412", "effective_tflops": "9.819047"},
        {"divergent_fraction": "0.2500", "avg_ms": "0.483226", "effective_tflops": "8.888120"},
        {"divergent_fraction": "0.3750", "avg_ms": "0.529469", "effective_tflops": "8.111832"},
        {"divergent_fraction": "0.5000", "avg_ms": "0.576266", "effective_tflops": "7.453095"},
        {"divergent_fraction": "0.6250", "avg_ms": "0.599532", "effective_tflops": "7.163872"},
        {"divergent_fraction": "0.7500", "avg_ms": "0.514826", "effective_tflops": "8.342557"},
        {"divergent_fraction": "0.8750", "avg_ms": "0.547983", "effective_tflops": "7.837770"},
        {"divergent_fraction": "1.0000", "avg_ms": "0.575939", "effective_tflops": "7.457336"},
    ],
    "02_coalescing_sweep": [
        {"stride": "1", "avg_ms": "0.107776", "effective_bw_gbps": "1245.339689"},
        {"stride": "2", "avg_ms": "0.152576", "effective_bw_gbps": "879.677821"},
        {"stride": "4", "avg_ms": "0.242893", "effective_bw_gbps": "552.580124"},
        {"stride": "8", "avg_ms": "0.432947", "effective_bw_gbps": "310.009456"},
        {"stride": "16", "avg_ms": "0.822886", "effective_bw_gbps": "163.106029"},
        {"stride": "32", "avg_ms": "0.841216", "effective_bw_gbps": "159.552040"},
        {"stride": "64", "avg_ms": "0.600320", "effective_bw_gbps": "223.576970"},
        {"stride": "128", "avg_ms": "0.237824", "effective_bw_gbps": "564.357348"},
    ],
    "03_shared_bank_conflict": [
        {"variant": "conflict", "avg_ms": "2.470025", "effective_bw_gbps": "869.417913"},
        {"variant": "padded", "avg_ms": "0.368981", "effective_bw_gbps": "5820.033361"},
    ],
    "04_launch_overhead": [
        {"mode": "many_launches", "batch": "1", "avg_ms": "0.003413"},
        {"mode": "fused", "batch": "1", "avg_ms": "0.003209"},
        {"mode": "many_launches", "batch": "10", "avg_ms": "0.031300"},
        {"mode": "fused", "batch": "10", "avg_ms": "0.003447"},
        {"mode": "many_launches", "batch": "50", "avg_ms": "0.155136"},
        {"mode": "fused", "batch": "50", "avg_ms": "0.004471"},
        {"mode": "many_launches", "batch": "100", "avg_ms": "0.307951"},
        {"mode": "fused", "batch": "100", "avg_ms": "0.005871"},
        {"mode": "many_launches", "batch": "250", "avg_ms": "0.763733"},
        {"mode": "fused", "batch": "250", "avg_ms": "0.009523"},
        {"mode": "many_launches", "batch": "500", "avg_ms": "1.526955"},
        {"mode": "fused", "batch": "500", "avg_ms": "0.015906"},
        {"mode": "many_launches", "batch": "1000", "avg_ms": "3.061351"},
        {"mode": "fused", "batch": "1000", "avg_ms": "0.028604"},
    ],
    "05_ilp_dependency": [
        {"variant": "dependent_chain", "avg_ms": "0.633830", "effective_tflops": "13.552418"},
        {"variant": "independent_ilp", "avg_ms": "0.594022", "effective_tflops": "14.460624"},
    ],
    "06_register_pressure": [
        {
            "variant": "low_register_pressure",
            "avg_ms": "0.594150",
            "core_effective_tflops": "14.457509",
            "theoretical_occupancy_pct": "100.00",
        },
        {
            "variant": "high_register_pressure",
            "avg_ms": "0.671002",
            "core_effective_tflops": "12.801661",
            "theoretical_occupancy_pct": "62.50",
        },
    ],
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.4, 5.4),
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.33,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#B8BEC4",
        }
    )


def find_latest_csv(result_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(result_dir.glob(f"{prefix}.*.csv"))
    return candidates[-1] if candidates else None


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _parse_float(value: str) -> float:
    text = value.strip()
    if "," in text and "." in text:
        text = text.replace(",", "")
    return float(text)


def _sorted_rows(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: _parse_float(row[key]))


def load_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_text(encoding="utf-8")
    lines = [
        line
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("==PROF==")
    ]
    if not lines:
        return []

    delimiter = _sniff_delimiter("\n".join(lines[:5]))
    reader = csv.DictReader(lines, delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for row in reader:
        clean = {k.strip(): (v.strip() if v is not None else "") for k, v in row.items() if k}
        # Drop index-like first column if someone exported with row numbers.
        if clean:
            keys = list(clean.keys())
            first_key = keys[0]
            if first_key in {"", "Unnamed: 0"} or first_key.lower().startswith("unnamed"):
                clean.pop(first_key, None)
        rows.append(clean)
    return rows


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_divergence(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = _sorted_rows(rows, "divergent_fraction")
    x = [_parse_float(r["divergent_fraction"]) for r in rows]
    y_tflops = [_parse_float(r["effective_tflops"]) for r in rows]
    baseline = y_tflops[0]
    y_norm = [v / baseline for v in y_tflops]

    fig, ax1 = plt.subplots()
    ax1.plot(x, y_tflops, marker="o", linewidth=2.6, color="#1f77b4", label="Throughput (TFLOP/s)")
    ax1.set_xlabel("Fraction of divergent warps")
    ax1.set_ylabel("Effective throughput [TFLOP/s]")
    ax1.set_title("Warp Divergence Sweep")
    ax1.set_xlim(-0.02, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(x, y_norm, marker="s", linewidth=2.0, color="#d62728", label="Normalized throughput")
    ax2.set_ylabel("Normalized throughput")
    ax2.set_ylim(0.55, 1.05)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save_plot(fig, out_path)


def plot_coalescing(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = _sorted_rows(rows, "stride")
    stride = [_parse_float(r["stride"]) for r in rows]
    ms = [_parse_float(r["avg_ms"]) for r in rows]
    bw = [_parse_float(r["effective_bw_gbps"]) for r in rows]
    bw_norm = [v / bw[0] for v in bw]

    fig, ax1 = plt.subplots()
    ax1.plot(stride, ms, marker="o", linewidth=2.6, color="#9467bd")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax1.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax1.set_xlabel("Stride")
    ax1.set_ylabel("Kernel time [ms]")
    ax1.set_title("Global-Memory Coalescing Sweep")

    ax2 = ax1.twinx()
    ax2.plot(stride, bw_norm, marker="s", linewidth=2.0, color="#2ca02c")
    ax2.set_ylabel("Normalized effective bandwidth proxy")
    ax2.set_ylim(bottom=0.0)
    ax1.axvspan(64, 128, color="#dddddd", alpha=0.3, zorder=0)
    save_plot(fig, out_path)


def plot_shared_bank_conflict(rows: list[dict[str, str]], out_path: Path) -> None:
    variants = [r["variant"] for r in rows]
    ms = [_parse_float(r["avg_ms"]) for r in rows]
    speedup = ms[0] / ms[1]

    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    bars = ax.bar(variants, ms, color=["#d62728", "#2ca02c"], width=0.6)
    ax.set_ylabel("Kernel time [ms]")
    ax.set_title("Shared-Memory Bank Conflict: Unpadded vs Padded")
    ax.text(
        0.5,
        max(ms) * 0.96,
        f"Padded speedup: {speedup:.2f}x",
        ha="center",
        va="top",
        fontsize=11.5,
        weight="bold",
    )
    for bar, value in zip(bars, ms):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.01, f"{value:.3f}", ha="center", va="bottom")
    save_plot(fig, out_path)


def plot_launch_overhead(rows: list[dict[str, str]], out_path: Path) -> None:
    many = _sorted_rows([r for r in rows if r["mode"] == "many_launches"], "batch")
    fused = _sorted_rows([r for r in rows if r["mode"] == "fused"], "batch")
    batch_many = [_parse_float(r["batch"]) for r in many]
    ms_many = [_parse_float(r["avg_ms"]) for r in many]
    batch_fused = [_parse_float(r["batch"]) for r in fused]
    ms_fused = [_parse_float(r["avg_ms"]) for r in fused]

    fig, ax = plt.subplots()
    ax.plot(batch_many, ms_many, marker="o", linewidth=2.6, color="#d62728", label="Many tiny launches")
    ax.plot(batch_fused, ms_fused, marker="s", linewidth=2.6, color="#1f77b4", label="Single fused kernel")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Equivalent operation count")
    ax.set_ylabel("Total time [ms]")
    ax.set_title("Kernel Launch Overhead and Fusion Benefit")
    ax.legend(loc="upper left")
    if batch_many and batch_fused and batch_many[-1] == batch_fused[-1]:
        tail_speedup = ms_many[-1] / ms_fused[-1]
        ax.text(
            batch_many[-1],
            ms_many[-1] * 0.85,
            f"{tail_speedup:.1f}x at batch={int(batch_many[-1])}",
            ha="right",
            va="top",
            fontsize=9.5,
            color="#333333",
        )
    save_plot(fig, out_path)


def plot_ilp_dependency(rows: list[dict[str, str]], out_path: Path) -> None:
    variants = [r["variant"] for r in rows]
    tflops = [_parse_float(r["effective_tflops"]) for r in rows]
    rel = tflops[1] / tflops[0]

    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    bars = ax.bar(variants, tflops, color=["#ff7f0e", "#2ca02c"], width=0.6)
    ax.set_ylabel("Effective throughput [TFLOP/s]")
    ax.set_title("Instruction-Level Parallelism: Dependent vs Independent")
    ax.text(
        0.5,
        max(tflops) * 0.98,
        f"ILP gain: {((rel - 1.0) * 100.0):.1f}%",
        ha="center",
        va="top",
        fontsize=11.5,
        weight="bold",
    )
    for bar, value in zip(bars, tflops):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.01, f"{value:.2f}", ha="center", va="bottom")
    save_plot(fig, out_path)


def plot_register_pressure(rows: list[dict[str, str]], out_path: Path) -> None:
    variants = [r["variant"] for r in rows]
    tflops = [_parse_float(r["core_effective_tflops"]) for r in rows]
    occ = [_parse_float(r["theoretical_occupancy_pct"]) for r in rows]
    x = range(len(variants))

    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
    ax1.bar([i - 0.18 for i in x], tflops, width=0.36, color="#1f77b4", label="Core effective TFLOP/s")
    ax1.set_ylabel("Core effective throughput [TFLOP/s]")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(variants, rotation=0)
    ax1.set_title("Register Pressure vs Throughput and Occupancy")

    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], occ, width=0.36, color="#ff7f0e", label="Theoretical occupancy (%)", alpha=0.9)
    ax2.set_ylabel("Theoretical occupancy [%]")
    ax2.set_ylim(0, 110)
    ax1.grid(False)
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    save_plot(fig, out_path)


def plot_arithmetic_intensity(rows: list[dict[str, str]], out_path: Path) -> None:
    rows = _sorted_rows(rows, "arithmetic_intensity_flop_per_byte")
    intensity = [_parse_float(r["arithmetic_intensity_flop_per_byte"]) for r in rows]
    tflops = [_parse_float(r["effective_tflops"]) for r in rows]
    bw = [_parse_float(r["effective_bw_gbps"]) for r in rows]

    fig, ax1 = plt.subplots()
    ax1.plot(intensity, tflops, marker="o", linewidth=2.6, color="#1f77b4", label="Throughput")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax1.set_ylabel("Effective throughput [TFLOP/s]")
    ax1.set_title("Arithmetic Intensity Sweep")

    ax2 = ax1.twinx()
    ax2.plot(intensity, bw, marker="s", linewidth=2.2, color="#2ca02c", label="Bandwidth proxy")
    ax2.set_ylabel("Effective bandwidth proxy [GB/s]")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    save_plot(fig, out_path)


PLOTTERS = {
    "01_divergence_sweep": plot_divergence,
    "02_coalescing_sweep": plot_coalescing,
    "03_shared_bank_conflict": plot_shared_bank_conflict,
    "04_launch_overhead": plot_launch_overhead,
    "05_ilp_dependency": plot_ilp_dependency,
    "06_register_pressure": plot_register_pressure,
    "07_arithmetic_intensity": plot_arithmetic_intensity,
}


def _iter_specs(specs: Iterable[DatasetSpec]) -> Iterable[DatasetSpec]:
    for spec in specs:
        yield spec


def _inline_rows(prefix: str) -> list[dict[str, str]] | None:
    rows = INLINE_FALLBACK_ROWS.get(prefix)
    if rows is None:
        return None
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory with results CSV files (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for output PDF plots (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    missing: list[str] = []
    used_inline: list[str] = []

    for spec in _iter_specs(SPECS):
        csv_path = find_latest_csv(args.input_dir, spec.prefix)
        rows: list[dict[str, str]] = []
        if csv_path is not None:
            rows = load_rows(csv_path)
        if not rows:
            inline_rows = _inline_rows(spec.prefix)
            if inline_rows is None:
                missing.append(spec.prefix)
                continue
            rows = inline_rows
            used_inline.append(spec.prefix)

        out_path = args.output_dir / spec.output_pdf
        plotter = PLOTTERS[spec.prefix]
        plotter(rows, out_path)
        generated.append(str(out_path.relative_to(ROOT)))

    if generated:
        print("Generated plots:")
        for path in generated:
            print(f"  {path}")
    if used_inline:
        print("Used inline fallback datasets:")
        for prefix in used_inline:
            print(f"  {prefix}")
    if missing:
        print("Missing datasets (no plot generated):")
        for prefix in missing:
            print(f"  {prefix}")


if __name__ == "__main__":
    main()
