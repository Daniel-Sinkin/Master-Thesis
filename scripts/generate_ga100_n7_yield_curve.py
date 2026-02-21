#!/usr/bin/env python3
"""Generate a simple defect-density yield curve for large GPU die areas.

Run from repository root:
    python3 scripts/generate_ga100_n7_yield_curve.py
"""

from __future__ import annotations

import math
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
IMAGES = ROOT / "images"

# Die areas in cm^2 (area-only sensitivity comparison).
GA100_AREA_CM2 = 8.26  # 826 mm^2 (A100 / GA100)
GV100_V100_AREA_CM2 = 8.15  # 815 mm^2 (V100 / GV100)
GH100_H200_AREA_CM2 = 8.14  # 814 mm^2 (H100/H200 / GH100)


def poisson_yield(area_cm2: float, defect_density_cm2: float) -> float:
    """First-order full-die yield model Y = exp(-A*D0)."""
    return math.exp(-area_cm2 * defect_density_cm2)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.8, 6.5),
            "figure.dpi": 140,
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def main() -> None:
    setup_style()
    IMAGES.mkdir(exist_ok=True)

    d0_values = [0.02 + i * 0.001 for i in range(381)]  # [0.02, 0.40]
    y_ga100 = [100.0 * poisson_yield(GA100_AREA_CM2, d0) for d0 in d0_values]
    y_v100 = [100.0 * poisson_yield(GV100_V100_AREA_CM2, d0) for d0 in d0_values]
    y_h200 = [100.0 * poisson_yield(GH100_H200_AREA_CM2, d0) for d0 in d0_values]

    # Reported N7 anchor points from industry coverage of TSMC/Intel disclosures.
    anchors = [
        ("Reported pre-MP reference", 0.33),
        ("Reported N7 at +3Q HVM", 0.09),
    ]

    fig, ax = plt.subplots()
    ax.plot(
        d0_values,
        y_ga100,
        color="#1f77b4",
        linewidth=2.8,
        label="GA100 area (826 mm$^2$)",
    )
    ax.plot(
        d0_values,
        y_v100,
        color="#ff7f0e",
        linewidth=2.2,
        linestyle="--",
        label="GV100 / V100 area (815 mm$^2$)",
    )
    ax.plot(
        d0_values,
        y_h200,
        color="#2ca02c",
        linewidth=2.2,
        linestyle="-.",
        label="GH100 / H200 area (814 mm$^2$)",
    )

    for idx, (label, d0) in enumerate(anchors):
        y = 100.0 * poisson_yield(GA100_AREA_CM2, d0)
        ax.axvline(d0, color="#6b7280", linestyle=(0, (3, 3)), linewidth=1.2)
        ax.scatter([d0], [y], color="#1f77b4", s=35, zorder=5)
        x_shift = 0.006 if idx == 0 else 0.003
        y_shift = -6.0 if idx == 0 else 5.0
        ax.text(
            d0 + x_shift,
            y + y_shift,
            f"{label}\n$D_0={d0:.2f}$, $Y={y:.1f}\\%$",
            fontsize=10,
            color="#111827",
        )

    ax.set_xlim(0.02, 0.40)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Defect density $D_0$ [defects/cm$^2$]")
    ax.set_ylabel("Predicted full-die yield [%]")
    ax.set_title("First-order Defect-Limited Yield Sensitivity by Large-Die Area")
    ax.legend(loc="upper right")

    fig.tight_layout()
    output = IMAGES / "ga100_n7_yield_curve.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
