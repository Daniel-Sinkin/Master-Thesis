#!/usr/bin/env python3
"""Extract median value of one metric from an Nsight Compute CSV log."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def parse_float(text: str) -> float:
    t = text.strip().replace('"', "")
    # Nsight CSV may emit thousands separators.
    if "," in t and "." in t:
        t = t.replace(",", "")
    return float(t)


def load_metric_values(path: Path, metric_name: str, kernel_substr: str) -> list[float]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith("==PROF==")]
    if not lines:
        return []

    reader = csv.DictReader(lines)
    values: list[float] = []
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        if name != metric_name:
            continue
        kernel = (row.get("Kernel Name") or "").strip()
        if kernel_substr and kernel_substr not in kernel:
            continue
        value_text = (row.get("Metric Value") or "").strip()
        if not value_text:
            continue
        values.append(parse_float(value_text))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="NCU CSV log file")
    parser.add_argument("--metric", required=True, help="Exact metric name")
    parser.add_argument(
        "--kernel-substr",
        default="hbm_stride_single_read_kernel",
        help="Only keep rows whose kernel name contains this substring",
    )
    args = parser.parse_args()

    vals = load_metric_values(args.input, args.metric, args.kernel_substr)
    if not vals:
        raise SystemExit(
            f"No rows found in {args.input} for metric '{args.metric}' "
            f"and kernel substring '{args.kernel_substr}'."
        )
    print(f"{statistics.median(vals):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
