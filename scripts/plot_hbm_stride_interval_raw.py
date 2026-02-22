#!/usr/bin/env python3
"""Plot HBM stride-interval raw benchmark CSV.

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
import statistics
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
DEFAULT_OUTPUT = ROOT / "images" / "hbm_stride_interval_raw.pdf"
STRIDES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


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


def plot(rows: list[dict[str, str]], output: Path, title: str, show_peak: bool) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        stride = int(float(row["stride_bytes"]))
        bw = float(row["achieved_bandwidth_gbps"])
        grouped[stride].append(bw)

    x_med: list[int] = []
    y_med: list[float] = []
    y_p10: list[float] = []
    y_p90: list[float] = []

    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=150)

    for stride in STRIDES:
        vals = sorted(grouped.get(stride, []))
        if not vals:
            continue
        x_raw = [stride] * len(vals)
        ax.scatter(x_raw, vals, s=18, alpha=0.35, color="#4C78A8", zorder=2)

        x_med.append(stride)
        y_med.append(statistics.median(vals))
        p10_idx = max(0, int(0.10 * (len(vals) - 1)))
        p90_idx = min(len(vals) - 1, int(0.90 * (len(vals) - 1)))
        y_p10.append(vals[p10_idx])
        y_p90.append(vals[p90_idx])

    if x_med:
        ax.plot(x_med, y_med, color="#1B7F3A", linewidth=2.4, marker="o", label="Median per stride", zorder=3)
        ax.fill_between(x_med, y_p10, y_p90, color="#1B7F3A", alpha=0.14, label="P10-P90 band", zorder=1)

    if show_peak:
        ax.axhline(1555.0, color="#B22222", linestyle="--", linewidth=1.8, label="A100 peak HBM: 1555 GB/s")

    ax.set_xscale("log", base=2)
    ax.set_xticks(STRIDES)
    ax.set_xticklabels([str(s) for s in STRIDES])
    ax.set_xlabel("Stride interval (bytes) between successive reads")
    ax.set_ylabel("Achieved bandwidth [GB/s]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.95)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot raw HBM stride benchmark CSV.")
    parser.add_argument("--input", type=Path, default=None, help="Input CSV path.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory to search latest CSV.")
    parser.add_argument("--prefix", type=str, default="08_hbm_stride_raw", help="Filename prefix for auto-discovery.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    parser.add_argument("--title", type=str, default="HBM Throughput vs Stride (8-byte reads, raw runs)")
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

    plot(rows, args.output, args.title, show_peak=not args.no_peak_line)
    print(f"Wrote plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
