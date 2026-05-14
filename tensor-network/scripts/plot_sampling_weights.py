#!/usr/bin/env python3
"""Plot PEPS sampling weights from the 2x2 amplitude CSV.

This intentionally uses only the Python standard library so the learning demo
does not depend on matplotlib being installed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def default_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    return (
        project_root / "peps_2x2_amplitudes.csv",
        project_root / "peps_2x2_sampling_weights.svg",
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} did not contain any rows")
    return rows


def write_svg(rows: list[dict[str, str]], path: Path) -> None:
    labels = [row["basis_state"] for row in rows]
    weights = [float(row["sampling_weight_percent"]) for row in rows]

    width = 1200
    height = 640
    margin_left = 72
    margin_right = 28
    margin_top = 44
    margin_bottom = 96
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_weight = max(weights)
    y_max = max(1.0, max_weight * 1.15)
    slot = plot_width / len(weights)
    bar_width = slot * 0.72

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1.0 - value / y_max)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#172026}'
        '.axis{stroke:#263238;stroke-width:1.2}'
        '.grid{stroke:#d7dde2;stroke-width:1}'
        '.bar{fill:#2f7f9f}'
        '.value{font-size:12px;text-anchor:middle}'
        '.label{font-size:12px;text-anchor:end}'
        '.tick{font-size:12px;text-anchor:end}'
        '.title{font-size:22px;font-weight:700;text-anchor:middle}'
        '</style>',
        f'<text class="title" x="{width / 2}" y="28">2x2 PEPS sampling weights p_Psi(S)</text>',
    ]

    tick_count = 5
    for tick in range(tick_count + 1):
        value = y_max * tick / tick_count
        y = y_pos(value)
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{margin_left - 10}" y="{y + 4:.2f}">{value:.1f}%</text>')

    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>')

    for idx, (label, weight) in enumerate(zip(labels, weights, strict=True)):
        x = margin_left + idx * slot + (slot - bar_width) / 2
        y = y_pos(weight)
        bar_height = height - margin_bottom - y
        parts.append(f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}"/>')
        parts.append(f'<text class="value" x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}">{weight:.2f}%</text>')
        label_x = x + bar_width / 2 + 4
        label_y = height - margin_bottom + 52
        parts.append(f'<text class="label" transform="rotate(-45 {label_x:.2f} {label_y:.2f})" x="{label_x:.2f}" y="{label_y:.2f}">{label}</text>')

    parts.append(f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" font-size="13">basis configuration S</text>')
    parts.append(f'<text transform="rotate(-90 18 {height / 2})" x="18" y="{height / 2}" text-anchor="middle" font-size="13">relative sampling weight</text>')
    parts.append("</svg>")

    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    default_csv, default_svg = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--out", type=Path, default=default_svg)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_svg(rows, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
