#!/usr/bin/env python3
"""Plot HBM stride-interval benchmark CSVs.

Expected CSV columns from phenomenon_hbm_stride_interval_raw:
  stride_bytes, run_idx, achieved_bandwidth_gbps, ...

Usage:
  python3 scripts/plot_hbm_stride_interval_raw.py \
    --input code/profiling/phenomena/results/08_hbm_stride_raw.csv \
    --output images/hbm_stride_interval_raw.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
DEFAULT_INPUT_DIR = ROOT / "code" / "profiling" / "phenomena" / "results"
DEFAULT_OUTPUT_MEDIAN = ROOT / "images" / "hbm_stride_interval_requested_median.pdf"
DEFAULT_OUTPUT_NCU = ROOT / "images" / "hbm_stride_interval_ncu_dram.pdf"
STRIDES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
TALK_STYLE_Y_TICKS = [128, 256, 512, 1024, 2048]
TALK_STYLE_Y_MIN = 64.0
TALK_STYLE_Y_MAX = 2048.0


def apply_talk_style_log_axes(ax: "plt.Axes") -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(STRIDES)
    ax.set_xticklabels([str(s) for s in STRIDES])

    ax.set_yscale("log", base=2)
    ax.set_ylim(TALK_STYLE_Y_MIN, TALK_STYLE_Y_MAX)
    ax.set_yticks(TALK_STYLE_Y_TICKS)
    ax.set_yticklabels([str(v) for v in TALK_STYLE_Y_TICKS])


def latest_csv(input_dir: Path, prefix: str) -> Path | None:
    files = sorted(input_dir.glob(f"{prefix}*.csv"))
    return files[-1] if files else None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            rows.append({k.strip(): (v.strip() if v is not None else "") for k, v in row.items()})
    return rows


def load_ncu_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: list[dict[str, str]] = []
        for row in reader:
            if row:
                out.append({k.strip(): (v.strip() if v is not None else "") for k, v in row.items()})
    return out


def _median(values: list[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        raise ValueError("Cannot compute median of empty list")
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def plot_requested_median(
    rows: list[dict[str, str]],
    output: Path,
    title: str,
) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        stride = int(float(row["stride_bytes"]))
        bw = float(row["achieved_bandwidth_gbps"])
        grouped[stride].append(bw)

    x_med: list[int] = []
    y_med: list[float] = []

    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=150)

    for stride in STRIDES:
        vals = sorted(grouped.get(stride, []))
        if not vals:
            continue
        x_med.append(stride)
        y_med.append(_median(vals))

    if x_med:
        ax.plot(x_med, y_med, color="#1B7F3A", linewidth=2.4, marker="o", label="Median per stride", zorder=3)

    apply_talk_style_log_axes(ax)
    ax.set_xlabel("Stride interval (bytes) between successive reads")
    ax.set_ylabel("Requested-read bandwidth [GB/s]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.95)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ncu_dram(
    ncu_rows: list[dict[str, str]],
    output: Path,
    title: str,
    show_peak: bool,
) -> None:
    ncu_map: dict[int, float] = {}
    for row in ncu_rows:
        stride = int(float(row["stride_bytes"]))
        gbps = float(row["dram_bandwidth_gbps"])
        ncu_map[stride] = gbps

    x = [s for s in STRIDES if s in ncu_map]
    y = [ncu_map[s] for s in x]
    if not x:
        raise SystemExit("NCU summary has no stride rows matching the expected stride set.")

    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=150)
    ax.plot(x, y, color="#D95F02", linewidth=2.3, marker="s", markersize=6, label="NCU DRAM throughput")

    if show_peak:
        ax.axhline(1555.0, color="#B22222", linestyle="--", linewidth=1.8, label="A100 peak HBM: 1555 GB/s")

    apply_talk_style_log_axes(ax)
    ax.set_xlabel("Stride interval (bytes) between successive reads")
    ax.set_ylabel("DRAM throughput [GB/s]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.95)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot HBM stride requested + NCU DRAM curves.")
    parser.add_argument("--input", type=Path, default=None, help="Input CSV path.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory to search latest CSV.")
    parser.add_argument("--prefix", type=str, default="08_hbm_stride_raw", help="Filename prefix for auto-discovery.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Deprecated alias for --output-median.",
    )
    parser.add_argument(
        "--output-median",
        type=Path,
        default=DEFAULT_OUTPUT_MEDIAN,
        help="Output PDF path for requested-bandwidth median plot.",
    )
    parser.add_argument(
        "--output-ncu",
        type=Path,
        default=DEFAULT_OUTPUT_NCU,
        help="Output PDF path for NCU DRAM-throughput plot.",
    )
    parser.add_argument(
        "--title-median",
        type=str,
        default="Requested Throughput vs Stride (8-byte reads, median over raw runs)",
    )
    parser.add_argument(
        "--title-ncu",
        type=str,
        default="NCU DRAM Throughput vs Stride (8-byte reads)",
    )
    parser.add_argument(
        "--ncu-summary",
        type=Path,
        default=None,
        help="Optional NCU summary CSV (stride_bytes, dram_bandwidth_gbps).",
    )
    parser.add_argument("--no-peak-line", action="store_true", help="Hide 1555 GB/s peak line.")
    args = parser.parse_args()

    input_csv = args.input
    if input_csv is None:
        input_csv = latest_csv(args.input_dir, args.prefix)
        if input_csv is None:
            raise SystemExit(
                f"No CSV found with prefix '{args.prefix}' in {args.input_dir}"
            )

    rows = load_rows(input_csv)
    if not rows:
        raise SystemExit(f"No data rows in {input_csv}")

    output_median = args.output if args.output is not None else args.output_median
    plot_requested_median(rows, output_median, args.title_median)
    print(f"Wrote plot: {output_median}")

    ncu_rows: list[dict[str, str]] | None = None
    if args.ncu_summary is not None:
        ncu_rows = load_ncu_rows(args.ncu_summary)
        if not ncu_rows:
            raise SystemExit(f"No rows in NCU summary CSV: {args.ncu_summary}")
        plot_ncu_dram(ncu_rows, args.output_ncu, args.title_ncu, show_peak=not args.no_peak_line)
        print(f"Wrote plot: {args.output_ncu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
