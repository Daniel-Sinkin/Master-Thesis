#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROFILING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROFILING_DIR.parent.parent
EXPERIMENTS_DIR = PROFILING_DIR / "experiments"
RESULTS_DIR = PROFILING_DIR / "results" / "experiments"
SLURM_RUNNER = PROFILING_DIR / "slurm" / "generic_experiment_profile.slurm"

DEFAULT_NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__thread_inst_executed_per_inst_executed.pct",
]


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env, check=check)


def b64(text: str) -> str:
    if not text:
        return ""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_space_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return shlex.split(value)


def load_sidecar_config(exp_path: Path) -> dict[str, Any]:
    cfg_path = exp_path.with_suffix(".json")
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {cfg_path}: {exc}") from exc


def list_experiments() -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(EXPERIMENTS_DIR.glob("*.cu"))


def resolve_experiment(expr: str) -> Path:
    candidate = Path(expr)
    if candidate.is_file() and candidate.suffix == ".cu":
        return candidate.resolve()

    if candidate.suffix != ".cu":
        named = EXPERIMENTS_DIR / f"{expr}.cu"
    else:
        named = EXPERIMENTS_DIR / candidate.name

    if named.exists():
        return named.resolve()

    known = "\n".join(f"  - {p.stem}" for p in list_experiments())
    raise SystemExit(f"Experiment '{expr}' not found. Known experiments:\n{known}")


def coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return shlex.split(value)
    raise SystemExit(f"Unsupported list value in config: {value!r}")


def coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def merged_setting(cli_value: Any, cfg: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def submit_experiment(exp_path: Path, args: argparse.Namespace) -> str:
    cfg = load_sidecar_config(exp_path)
    exp_name = exp_path.stem

    profile = coerce_str(merged_setting(args.profile, cfg, "profile", "none"), "none")

    run_args_val = merged_setting(args.run_args, cfg, "run_args", [])
    run_args = parse_space_list(run_args_val) if isinstance(run_args_val, str) else coerce_list(run_args_val)

    libs_val = merged_setting(args.libs, cfg, "libs", ["cudart"])
    libs = parse_csv(libs_val) if isinstance(libs_val, str) else coerce_list(libs_val)
    if not libs:
        libs = ["cudart"]

    nvcc_flags_val = merged_setting(args.nvcc_flags, cfg, "nvcc_flags", [])
    nvcc_flags = parse_space_list(nvcc_flags_val) if isinstance(nvcc_flags_val, str) else coerce_list(nvcc_flags_val)

    ncu_metrics_val = merged_setting(args.ncu_metrics, cfg, "ncu_metrics", DEFAULT_NCU_METRICS)
    ncu_metrics = parse_csv(ncu_metrics_val) if isinstance(ncu_metrics_val, str) else coerce_list(ncu_metrics_val)
    if not ncu_metrics:
        ncu_metrics = DEFAULT_NCU_METRICS

    settings = {
        "account": coerce_str(merged_setting(args.account, cfg, "account", "slai"), "slai"),
        "partition": coerce_str(merged_setting(args.partition, cfg, "partition", "dc-gpu-devel"), "dc-gpu-devel"),
        "time": coerce_str(merged_setting(args.time, cfg, "time", "00:20:00"), "00:20:00"),
        "cpus_per_task": coerce_int(merged_setting(args.cpus_per_task, cfg, "cpus_per_task", 12), 12),
        "gpus": coerce_int(merged_setting(args.gpus, cfg, "gpus", 1), 1),
        "arch": coerce_int(merged_setting(args.arch, cfg, "arch", 80), 80),
        "nsys_trace": coerce_str(merged_setting(args.nsys_trace, cfg, "nsys_trace", "cuda,osrt"), "cuda,osrt"),
        "nsys_sample": coerce_str(merged_setting(args.nsys_sample, cfg, "nsys_sample", "none"), "none"),
        "ncu_kernel": coerce_str(merged_setting(args.ncu_kernel, cfg, "ncu_kernel", ""), ""),
        "ncu_launch_skip": coerce_int(merged_setting(args.ncu_launch_skip, cfg, "ncu_launch_skip", 0), 0),
        "ncu_launch_count": coerce_int(merged_setting(args.ncu_launch_count, cfg, "ncu_launch_count", 1), 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "EXPERIMENT_CU": str(exp_path),
            "EXPERIMENT_NAME": exp_name,
            "PROFILE_RESULTS_ROOT": str(RESULTS_DIR),
            "PROFILE_MODE": profile,
            "CUDA_ARCH": str(settings["arch"]),
            "RUN_ARGS_B64": b64(" ".join(run_args)),
            "LIBS_CSV": ",".join(libs),
            "EXTRA_NVCC_FLAGS_B64": b64(" ".join(nvcc_flags)),
            "NCU_METRICS_B64": b64(",".join(ncu_metrics)),
            "NSYS_TRACE": settings["nsys_trace"],
            "NSYS_SAMPLE": settings["nsys_sample"],
            "NCU_KERNEL_REGEX": settings["ncu_kernel"],
            "NCU_LAUNCH_SKIP": str(settings["ncu_launch_skip"]),
            "NCU_LAUNCH_COUNT": str(settings["ncu_launch_count"]),
        }
    )

    job_name = f"prof_{exp_name}"[:90]
    cmd = [
        "sbatch",
        "--parsable",
        "--chdir",
        str(REPO_ROOT),
        "--job-name",
        job_name,
        "--account",
        settings["account"],
        "--partition",
        settings["partition"],
        "--gres",
        f"gpu:{settings['gpus']}",
        "--cpus-per-task",
        str(settings["cpus_per_task"]),
        "--time",
        settings["time"],
        "--output",
        str(RESULTS_DIR / f"{exp_name}_slurm-%j.out"),
        "--error",
        str(RESULTS_DIR / f"{exp_name}_slurm-%j.err"),
        str(SLURM_RUNNER),
    ]

    if args.dry_run:
        print("Dry-run sbatch command:")
        print(" ".join(shlex.quote(part) for part in cmd))
        return "0"

    proc = run_cmd(cmd, env=env, check=True)
    job_id = proc.stdout.strip().splitlines()[-1]

    local_latest = RESULTS_DIR / f"{exp_name}_latest_submitted.txt"
    local_latest.write_text(f"{job_id}\n", encoding="utf-8")

    print(f"Submitted {exp_name} as job {job_id}")
    print(f"Track: squeue -j {job_id} -o \"%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R\"")
    print(f"Logs:  {RESULTS_DIR}/{exp_name}_slurm-{job_id}.out")
    print(f"Run:   {RESULTS_DIR}/job_{job_id}/{exp_name}_run.log")
    print("Pull:  python code/profiling/orchestrate.py pull <user@cluster-login> "
          f"{exp_name} {job_id}")
    return job_id


def cmd_init_venv(args: argparse.Namespace) -> int:
    venv_dir = (REPO_ROOT / args.venv).resolve()
    python_bin = args.python

    if venv_dir.exists() and not args.force:
        print(f"Virtual environment already exists at {venv_dir}")
        print("Use --force to recreate it.")
        return 0

    if venv_dir.exists() and args.force:
        shutil.rmtree(venv_dir)

    subprocess.run([python_bin, "-m", "venv", str(venv_dir)], check=True)
    pip_path = venv_dir / "bin" / "pip"
    subprocess.run([str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)

    print(f"Created virtual environment: {venv_dir}")
    print(f"Activate: source {venv_dir}/bin/activate")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    experiments = list_experiments()
    if not experiments:
        print(f"No experiments found in {EXPERIMENTS_DIR}")
        return 0

    print(f"Experiments in {EXPERIMENTS_DIR}:")
    for path in experiments:
        cfg = path.with_suffix(".json")
        suffix = " (config)" if cfg.exists() else ""
        print(f"  - {path.stem}{suffix}")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    stem = args.name.replace(".cu", "")
    cu_path = EXPERIMENTS_DIR / f"{stem}.cu"
    cfg_path = EXPERIMENTS_DIR / f"{stem}.json"

    if (cu_path.exists() or cfg_path.exists()) and not args.force:
        raise SystemExit(f"{stem} already exists. Use --force to overwrite.")

    cu_source = f"""#include <cuda_runtime.h>\n#include <iostream>\n\n__global__ void {stem}_kernel(float* data, int n) {{\n  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n  if (idx < n) {{\n    data[idx] = data[idx] * 2.0f + 1.0f;\n  }}\n}}\n\nint main(int argc, char** argv) {{\n  int n = 1 << 20;\n  if (argc > 1) n = std::atoi(argv[1]);\n\n  float* d = nullptr;\n  cudaMalloc(&d, n * sizeof(float));\n  cudaMemset(d, 0, n * sizeof(float));\n\n  int threads = 256;\n  int blocks = (n + threads - 1) / threads;\n  {stem}_kernel<<<blocks, threads>>>(d, n);\n  cudaDeviceSynchronize();\n\n  cudaFree(d);\n  std::cout << \"ok,n=\" << n << '\\n';\n  return 0;\n}}\n"""

    default_cfg = {
        "profile": "none",
        "libs": ["cudart"],
        "run_args": ["1048576"],
        "nvcc_flags": [],
        "arch": 80,
        "account": "slai",
        "partition": "dc-gpu-devel",
        "time": "00:20:00",
        "cpus_per_task": 12,
        "gpus": 1,
        "nsys_trace": "cuda,osrt",
        "nsys_sample": "none",
        "ncu_metrics": DEFAULT_NCU_METRICS,
        "ncu_kernel": "",
        "ncu_launch_skip": 0,
        "ncu_launch_count": 1,
    }

    cu_path.write_text(cu_source, encoding="utf-8")
    cfg_path.write_text(json.dumps(default_cfg, indent=2) + "\n", encoding="utf-8")

    print(f"Created {cu_path}")
    print(f"Created {cfg_path}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    if not SLURM_RUNNER.exists():
        raise SystemExit(f"Missing SLURM runner: {SLURM_RUNNER}")

    if args.experiment == "all":
        experiments = list_experiments()
        if not experiments:
            raise SystemExit(f"No .cu files found in {EXPERIMENTS_DIR}")
        for exp in experiments:
            submit_experiment(exp, args)
        return 0

    exp_path = resolve_experiment(args.experiment)
    submit_experiment(exp_path, args)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    experiment = args.experiment.replace(".cu", "")
    remote_repo_root = args.remote_repo_root
    remote_results = f"{remote_repo_root}/code/profiling/results/experiments"

    job_id = args.job_id
    if job_id == "latest":
        query = [
            "ssh",
            args.remote_login,
            f"cat '{remote_results}/{experiment}_latest_job_id.txt'",
        ]
        proc = run_cmd(query, check=True)
        job_id = proc.stdout.strip()
        if not job_id:
            raise SystemExit(
                f"Could not resolve latest job id for {experiment} from {remote_results}."
            )

    local_dest = Path(args.local_dest) if args.local_dest else RESULTS_DIR
    local_dest.mkdir(parents=True, exist_ok=True)

    remote_job_dir = f"{remote_results}/job_{job_id}/"
    remote_tar = f"{remote_results}/{experiment}_job_{job_id}.tar.gz"

    if shutil.which("rsync"):
        subprocess.run(
            ["rsync", "-avz", f"{args.remote_login}:{remote_job_dir}", f"{local_dest}/job_{job_id}/"],
            check=True,
        )
        subprocess.run(
            ["rsync", "-avz", f"{args.remote_login}:{remote_tar}", f"{local_dest}/"],
            check=False,
        )
    else:
        (local_dest / f"job_{job_id}").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["scp", "-r", f"{args.remote_login}:{remote_job_dir}.", f"{local_dest}/job_{job_id}/"],
            check=True,
        )
        subprocess.run(
            ["scp", f"{args.remote_login}:{remote_tar}", f"{local_dest}/"],
            check=False,
        )

    print(f"Pulled run {job_id} for {experiment} into {local_dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA experiment orchestration for pod profiling workflows"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_venv = sub.add_parser("init-venv", help="Create local .venv for orchestration tooling")
    p_venv.add_argument("--python", default="python3.12", help="Python executable (default: python3.12)")
    p_venv.add_argument("--venv", default=".venv", help="Path to venv relative to repo root")
    p_venv.add_argument("--force", action="store_true", help="Recreate venv if it already exists")
    p_venv.set_defaults(func=cmd_init_venv)

    p_list = sub.add_parser("list", help="List available .cu experiments")
    p_list.set_defaults(func=cmd_list)

    p_new = sub.add_parser("new", help="Create a new experiment .cu + .json scaffold")
    p_new.add_argument("name", help="Experiment base name")
    p_new.add_argument("--force", action="store_true", help="Overwrite if files already exist")
    p_new.set_defaults(func=cmd_new)

    p_submit = sub.add_parser("submit", help="Submit one experiment or all experiments via SLURM")
    p_submit.add_argument("experiment", help="Experiment name/path or 'all'")
    p_submit.add_argument("--profile", choices=["none", "nsys", "ncu", "both"], default=None)
    p_submit.add_argument("--run-args", default=None, help="Program arguments, e.g. '128 47 84 64 4 24'")
    p_submit.add_argument("--libs", default=None, help="CUDA libs CSV, e.g. 'cudart,cublas'")
    p_submit.add_argument("--nvcc-flags", default=None, help="Extra NVCC flags, e.g. '-I/path -DUSE_X=1'")

    p_submit.add_argument("--account", default=None)
    p_submit.add_argument("--partition", default=None)
    p_submit.add_argument("--time", default=None)
    p_submit.add_argument("--cpus-per-task", type=int, default=None)
    p_submit.add_argument("--gpus", type=int, default=None)
    p_submit.add_argument("--arch", type=int, default=None)

    p_submit.add_argument("--nsys-trace", default=None)
    p_submit.add_argument("--nsys-sample", default=None)
    p_submit.add_argument("--ncu-metrics", default=None, help="NCU metrics CSV override")
    p_submit.add_argument("--ncu-kernel", default=None, help="NCU -k kernel regex")
    p_submit.add_argument("--ncu-launch-skip", type=int, default=None)
    p_submit.add_argument("--ncu-launch-count", type=int, default=None)
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.set_defaults(func=cmd_submit)

    p_pull = sub.add_parser("pull", help="Fetch results from pod/login node")
    p_pull.add_argument("remote_login", help="Remote login, e.g. sinkin1@jrc0225")
    p_pull.add_argument("experiment", help="Experiment base name")
    p_pull.add_argument("job_id", nargs="?", default="latest", help="Job id or 'latest'")
    p_pull.add_argument(
        "remote_repo_root",
        nargs="?",
        default="/p/home/jusers/sinkin1/jureca/Master-Thesis",
        help="Remote repository root",
    )
    p_pull.add_argument("local_dest", nargs="?", default=None, help="Local destination folder")
    p_pull.set_defaults(func=cmd_pull)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
