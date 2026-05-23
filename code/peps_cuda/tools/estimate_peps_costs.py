#!/usr/bin/env python3
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Gpu:
    name: str
    fp64_tflops: float
    fp64_tc_tflops: float
    fp32_tflops: float
    mem_tb_s: float
    hbm_gb: float
    sm_count: int


GPUS = {
    "a100_40": Gpu("A100 SXM 40GB", 9.7, 19.5, 19.5, 1.555, 40.0, 108),
    "h100_80": Gpu("H100 SXM 80GB", 34.0, 67.0, 67.0, 3.35, 80.0, 132),
    "h200_141": Gpu("H200 SXM 141GB", 34.0, 67.0, 67.0, 4.8, 141.0, 132),
    "jupiter_gh200": Gpu("JUPITER GH200 H100 96GB", 34.0, 67.0, 67.0, 4.0, 96.0, 132),
}


def single_layer_step_flops(dc: int, d: int) -> float:
    return (dc**3) * (d**3) + (dc**2) * (d**4)


def double_layer_step_flops(dc: int, d: int, local_dim: int) -> float:
    return (dc**3) * (d**4) + local_dim * (dc**2) * (d**6)


def format_bytes(nbytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(nbytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=GPUS, default="jupiter_gh200")
    parser.add_argument("--lx", type=int, default=16)
    parser.add_argument("--ly", type=int, default=16)
    parser.add_argument("--d", type=int, default=8, help="PEPS bond dimension D")
    parser.add_argument("--dc", type=int, default=64, help="boundary MPS bond dimension")
    parser.add_argument("--local-dim", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    gpu = GPUS[args.gpu]
    row_steps = max(args.lx - 1, 1) * args.ly
    single = row_steps * single_layer_step_flops(args.dc, args.d)
    double = row_steps * double_layer_step_flops(args.dc, args.d, args.local_dim)
    sample_single = args.samples * single
    bulk_parameter_count = args.lx * args.ly * args.local_dim * (args.d**4)
    sampled_sector_parameter_count = args.lx * args.ly * (args.d**4)
    dense_o_bytes = args.samples * bulk_parameter_count * 16
    sampled_sector_o_bytes = args.samples * sampled_sector_parameter_count * 16
    gram_bytes = args.samples * args.samples * 16
    dense_gram_dot_elems_per_pair = bulk_parameter_count
    sampled_sector_dot_elems_per_pair = sampled_sector_parameter_count
    random_direct_dot_elems_per_pair = (
        sampled_sector_parameter_count / args.local_dim
    )
    direct_vs_dense_ratio = (
        random_direct_dot_elems_per_pair / dense_gram_dot_elems_per_pair
        if dense_gram_dot_elems_per_pair
        else 0.0
    )

    print(f"GPU: {gpu.name}")
    print(f"Lattice: {args.lx}x{args.ly}, D={args.d}, Dc={args.dc}, d={args.local_dim}")
    print(f"Approx bulk parameters:           {bulk_parameter_count:,}")
    print(f"Dense O storage:                  {format_bytes(dense_o_bytes)}")
    print(f"Sampled-sector O storage:         {format_bytes(sampled_sector_o_bytes)}")
    print(f"Sample Gram storage:              {format_bytes(gram_bytes)}")
    print(f"Dense Gram dot elems/pair:        {dense_gram_dot_elems_per_pair:,}")
    print(f"Sampled-sector elems/pair:        {sampled_sector_dot_elems_per_pair:,}")
    print(
        "Random direct Gram elems/pair:    "
        f"{random_direct_dot_elems_per_pair:,.0f}"
    )
    print(f"Random direct/dense dot ratio:    {direct_vs_dense_ratio:.4f}")
    print(f"Single-layer boundary work/sample: {single/1e12:.4f} TFLOP-ish")
    print(f"All sample single-layer work:       {sample_single/1e12:.4f} TFLOP-ish")
    print(f"Double-layer boundary refresh:      {double/1e12:.4f} TFLOP-ish")
    print()
    print("Ideal lower bounds if compute-bound:")
    print(f"  single samples @ FP64 CUDA cores: {sample_single/(gpu.fp64_tflops*1e12):.4f} s")
    print(f"  double refresh @ FP64 CUDA cores: {double/(gpu.fp64_tflops*1e12):.4f} s")
    print(f"  double refresh @ FP64 tensorcore: {double/(gpu.fp64_tc_tflops*1e12):.4f} s")
    print()
    print("Use this only for order-of-magnitude triage. The real bottleneck is likely")
    print("small/irregular GEMM launch count, truncation/SVD traffic, and boundary reuse.")


if __name__ == "__main__":
    main()
