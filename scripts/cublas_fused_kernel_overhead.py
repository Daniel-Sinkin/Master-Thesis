# scripts/cublas_fused_kernel_overhead.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ns = [1, 2, 4, 8, 13, 16, 23, 32]

separate_total = [
    0.00802182 * 1000,
    0.00794095 * 1000,
    0.00783545 * 1000,
    0.00789512 * 1000,
    0.00847657 * 1000,
    0.00810371 * 1000,
    0.0092168 * 1000,
    0.00975229 * 1000,
]

fused_total = [
    0.171936,
    0.184544,
    0.180064,
    0.180064,
    0.354656,
    0.347008,
    0.717056,
    1.37968,
]

speedups = [s / f for s, f in zip(separate_total, fused_total)]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(
        ns,
        separate_total,
        "s-",
        color="tab:red",
        markersize=6,
        label="1000 separate kernel launches",
    )
    ax.semilogy(
        ns,
        fused_total,
        "o-",
        color="tab:blue",
        markersize=6,
        label="1 fused kernel (1000 operations)",
    )

    for n, sep, fused, speedup in zip(ns, separate_total, fused_total, speedups):
        mid_y = np.sqrt(sep * fused)
        ax.text(
            n,
            mid_y,
            f"{speedup:.0f}×",
            fontsize=8,
            color="#444444",
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_xlabel("Matrix size n (n × n GEMM, FP64)")
    ax.set_ylabel("Total time (ms)")
    ax.set_xticks(ns)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("images/kernel_fusion_benchmark.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
