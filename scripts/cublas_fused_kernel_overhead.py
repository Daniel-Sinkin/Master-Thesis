# scripts/cublas_fused_kernel_overhead.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

dims = [1, 2, 4, 8, 13, 16, 23, 32]

# Times in microseconds
separate_total = [
    0.00802182 * 1e6,
    0.00794095 * 1e6,
    0.00783545 * 1e6,
    0.00789512 * 1e6,
    0.00847657 * 1e6,
    0.00810371 * 1e6,
    0.0092168 * 1e6,
    0.00975229 * 1e6,
]

fused_total = [
    0.171936 * 1e3,
    0.184544 * 1e3,
    0.180064 * 1e3,
    0.180064 * 1e3,
    0.354656 * 1e3,
    0.347008 * 1e3,
    0.717056 * 1e3,
    1.37968 * 1e3,
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.semilogy(
        dims,
        separate_total,
        "s-",
        color="tab:red",
        markersize=6,
        label="1000 separate kernel launches",
    )
    ax.semilogy(
        dims,
        fused_total,
        "o-",
        color="tab:blue",
        markersize=6,
        label="1 fused kernel (1000 operations)",
    )

    ax.set_xlabel("Matrix size n (n x n GEMM, FP64)")
    ax.set_ylabel("Total time (µs)")
    ax.set_xticks(dims)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("images/kernel_fusion_benchmark.svg", format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
