#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def fmt_complex(pair):
    return f"{float(pair[0]):+.6g}{float(pair[1]):+.6g}i"


def first_line(value: str, limit: int = 96) -> str:
    line = value.splitlines()[0] if value else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    for line in args.path.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") != "logpsi":
            continue
        sample = row.get("sample_row_major", row.get("sample", []))
        env_status = "env-ok" if "logpsi_env" in row else "env-error"
        ok_status = (
            f"Ok={row['ok_length']} norm2={float(row['ok_norm2']):.6g}"
            if "ok_length" in row
            else "Ok-missing"
        )
        if "e_o_error" in row:
            ok_status += f" e/o-error={first_line(row['e_o_error'])}"
        if "logpsi_env_error" in row:
            env_status = f"env-error={first_line(row['logpsi_env_error'])}"
        print(
            f"{row['name']}: L={row['lx']}x{row['ly']} D={row['bond_dim']} "
            f"type={row['type']} sample={sample} "
            f"logpsi={fmt_complex(row['logpsi_exact'])} {env_status} {ok_status}"
        )


if __name__ == "__main__":
    main()
