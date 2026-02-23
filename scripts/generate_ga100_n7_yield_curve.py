#!/usr/bin/env python3
"""Generate defect-density yield curves for GA100 under simple harvest models.

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

# Die area in cm^2.
GA100_AREA_CM2 = 8.26  # 826 mm^2 (A100 / GA100)
GA100_SM_PARTITIONS = 128


def poisson_yield(area_cm2: float, defect_density_cm2: float) -> float:
    """First-order full-die yield model Y = exp(-A*D0)."""
    return math.exp(-area_cm2 * defect_density_cm2)


def yield_at_most_k_broken_partitions(
    area_cm2: float, defect_density_cm2: float, partitions: int, k: int
) -> float:
    """Yield for at most k broken partitions under an independent-area model.

    Assumption: the die is split into equal critical partitions of area A/N.
    For each partition, survival probability is q = exp(-(A/N)D0), and broken
    probability is p = 1-q. The number of broken partitions K follows
    Binomial(N, p), so Y_{<=k} = P(K <= k).
    """
    area_per_partition = area_cm2 / partitions
    q = math.exp(-area_per_partition * defect_density_cm2)
    p = 1.0 - q
    return sum(
        math.comb(partitions, i) * (p**i) * (q ** (partitions - i))
        for i in range(k + 1)
    )


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
    y_full = [100.0 * poisson_yield(GA100_AREA_CM2, d0) for d0 in d0_values]
    y_le1 = [
        100.0
        * yield_at_most_k_broken_partitions(
            GA100_AREA_CM2, d0, GA100_SM_PARTITIONS, 1
        )
        for d0 in d0_values
    ]
    y_le2 = [
        100.0
        * yield_at_most_k_broken_partitions(
            GA100_AREA_CM2, d0, GA100_SM_PARTITIONS, 2
        )
        for d0 in d0_values
    ]

    # Reported N7 anchor points from industry coverage of TSMC/Intel disclosures.
    anchors = [0.33, 0.09]

    fig, ax = plt.subplots()
    ax.plot(d0_values, y_full, color="#1f77b4", linewidth=2.8, label="0 broken (full die)")
    ax.plot(
        d0_values,
        y_le1,
        color="#ff7f0e",
        linewidth=2.4,
        label=r"$\leq$1 broken partition (out of 128)",
    )
    ax.plot(
        d0_values,
        y_le2,
        color="#2ca02c",
        linewidth=2.4,
        label=r"$\leq$2 broken partitions (out of 128)",
    )
    for d0 in anchors:
        y0 = 100.0 * poisson_yield(GA100_AREA_CM2, d0)
        y1 = (
            100.0
            * yield_at_most_k_broken_partitions(
                GA100_AREA_CM2, d0, GA100_SM_PARTITIONS, 1
            )
        )
        y2 = (
            100.0
            * yield_at_most_k_broken_partitions(
                GA100_AREA_CM2, d0, GA100_SM_PARTITIONS, 2
            )
        )
        ax.scatter([d0], [y0], color="#1f77b4", edgecolors="#0f172a", linewidths=0.6, s=34, zorder=5)
        ax.scatter([d0], [y1], color="#ff7f0e", edgecolors="#0f172a", linewidths=0.6, s=34, zorder=5)
        ax.scatter([d0], [y2], color="#2ca02c", edgecolors="#0f172a", linewidths=0.6, s=34, zorder=5)

    ax.set_xlim(0.02, 0.40)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Defect density $D_0$ [defects/cm$^2$]")
    ax.set_ylabel("Predicted yield [%]")
    ax.set_title("GA100 Yield Sensitivity with Simple Harvest Allowance")
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    output = IMAGES / "ga100_n7_yield_curve.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
