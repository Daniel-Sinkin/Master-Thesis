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
    curves = [
        ("0 broken (full die)", 0, "#1f77b4", 2.8),
        (r"$\leq$1 broken partition (out of 128)", 1, "#ff7f0e", 2.2),
        (r"$\leq$2 broken partitions (out of 128)", 2, "#2ca02c", 2.2),
        (r"$\leq$3 broken partitions (out of 128)", 3, "#d62728", 2.2),
        (r"$\leq$4 broken partitions (out of 128)", 4, "#9467bd", 2.2),
        (r"$\leq$20 broken partitions (A100 108/128 case)", 20, "#8c564b", 2.6),
    ]

    # Reported N7 anchor points from industry coverage of TSMC/Intel disclosures.
    anchors = [0.33, 0.09]

    fig, ax = plt.subplots()
    for label, k, color, lw in curves:
        if k == 0:
            y = [100.0 * poisson_yield(GA100_AREA_CM2, d0) for d0 in d0_values]
        else:
            y = [
                100.0
                * yield_at_most_k_broken_partitions(
                    GA100_AREA_CM2, d0, GA100_SM_PARTITIONS, k
                )
                for d0 in d0_values
            ]
        ax.plot(d0_values, y, color=color, linewidth=lw, label=label)
        for d0 in anchors:
            if k == 0:
                y_anchor = 100.0 * poisson_yield(GA100_AREA_CM2, d0)
            else:
                y_anchor = (
                    100.0
                    * yield_at_most_k_broken_partitions(
                        GA100_AREA_CM2, d0, GA100_SM_PARTITIONS, k
                    )
                )
            ax.scatter(
                [d0],
                [y_anchor],
                color=color,
                edgecolors="#0f172a",
                linewidths=0.55,
                s=28,
                zorder=5,
            )

    ax.set_xlim(0.02, 0.40)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Defect density $D_0$ [defects/cm$^2$]")
    ax.set_ylabel("Predicted yield [%]")
    ax.set_title("GA100 Yield Sensitivity with Harvest Allowance (k=0..4 and k=20)")
    ax.legend(loc="center right", fontsize=9.3)

    fig.tight_layout()
    output = IMAGES / "ga100_n7_yield_curve.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
