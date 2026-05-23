#!/usr/bin/env python3
import argparse
import itertools
import json
import math
from pathlib import Path


def complex_pair(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"expected [real, imag], got {value!r}")
    return complex(float(value[0]), float(value[1]))


def flat_index(indices, dims):
    offset = 0
    stride = 1
    for index, dim in zip(indices, dims):
        offset += int(index) * stride
        stride *= int(dim)
    return offset


def site_offsets(site_dims):
    offsets = []
    offset = 0
    for dims in site_dims:
        offsets.append(offset)
        count = 1
        for dim in dims:
            count *= int(dim)
        offset += count
    return offsets


def exact_amplitude_from_fixture(row, theta, site_dims):
    axes = row["theta_axis_labels"]
    offsets = site_offsets(site_dims)
    sample = row.get("sample_row_major", row["sample"])

    link_dims = {}
    for labels, dims in zip(axes, site_dims):
        for label, dim in zip(labels[1:], dims[1:]):
            previous = link_dims.setdefault(label, int(dim))
            if previous != int(dim):
                raise ValueError(f"link dimension mismatch for {label}")

    labels = list(link_dims)
    ranges = [range(link_dims[label]) for label in labels]
    amplitude = 0.0 + 0.0j
    for assignment_tuple in itertools.product(*ranges):
        assignment = dict(zip(labels, assignment_tuple))
        term = 1.0 + 0.0j
        for site, (site_axes, dims, offset) in enumerate(
            zip(axes, site_dims, offsets)
        ):
            indices = [int(sample[site])]
            indices.extend(assignment[label] for label in site_axes[1:])
            term *= theta[offset + flat_index(indices, dims)]
        amplitude += term
    return amplitude


def exact_log_gradients_from_fixture(row, theta, site_dims, amplitude):
    axes = row["theta_axis_labels"]
    offsets = site_offsets(site_dims)
    sample = row.get("sample_row_major", row["sample"])

    link_dims = {}
    for labels, dims in zip(axes, site_dims):
        for label, dim in zip(labels[1:], dims[1:]):
            link_dims.setdefault(label, int(dim))

    labels = list(link_dims)
    ranges = [range(link_dims[label]) for label in labels]
    gradients = [0.0 + 0.0j for _ in theta]
    for assignment_tuple in itertools.product(*ranges):
        assignment = dict(zip(labels, assignment_tuple))
        values = []
        local_indices = []
        for site, (site_axes, dims, offset) in enumerate(
            zip(axes, site_dims, offsets)
        ):
            indices = [int(sample[site])]
            indices.extend(assignment[label] for label in site_axes[1:])
            local = flat_index(indices, dims)
            local_indices.append(offset + local)
            values.append(theta[offset + local])
        for site, parameter in enumerate(local_indices):
            product = 1.0 + 0.0j
            for other, value in enumerate(values):
                if other != site:
                    product *= value
            gradients[parameter] += product
    return [value / amplitude for value in gradients]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.path.read_text().splitlines()]
    if not rows:
        raise SystemExit("empty fixture file")
    if rows[0].get("kind") != "metadata":
        raise SystemExit("first fixture row must be metadata")

    logpsi_rows = [row for row in rows if row.get("kind") == "logpsi"]
    if not logpsi_rows:
        raise SystemExit("no logpsi fixture rows")

    max_env_error = 0.0
    boundary_errors = 0
    reconstructed_rows = 0
    for row in logpsi_rows:
        exact = complex_pair(row["logpsi_exact"])
        if not math.isfinite(exact.real) or not math.isfinite(exact.imag):
            raise SystemExit(f"non-finite exact logpsi in {row['name']}")
        if "theta" in row:
            theta = [complex_pair(value) for value in row["theta"]]
            if len(theta) != int(row["theta_length"]):
                raise SystemExit(f"theta length mismatch in {row['name']}")
            site_dims = row.get("theta_site_dims", row.get("site_dims"))
            if site_dims is not None:
                parameter_count = 0
                for dims in site_dims:
                    count = 1
                    for dim in dims:
                        count *= int(dim)
                    parameter_count += count
                if parameter_count != len(theta):
                    raise SystemExit(f"site dims/theta mismatch in {row['name']}")
            if "theta_axis_labels" in row:
                if len(row["theta_axis_labels"]) != int(row["lx"]) * int(row["ly"]):
                    raise SystemExit(f"theta axis site count mismatch in {row['name']}")
                for labels, dims in zip(row["theta_axis_labels"], site_dims):
                    if len(labels) != len(dims):
                        raise SystemExit(f"theta axis rank mismatch in {row['name']}")
                if "sample" in row and len(theta) <= 4096:
                    amplitude = exact_amplitude_from_fixture(row, theta, site_dims)
                    exact_log = complex_pair(row["logpsi_exact"])
                    reconstructed_log = complex(
                        math.log(abs(amplitude)),
                        math.atan2(amplitude.imag, amplitude.real),
                    )
                    if abs(reconstructed_log - exact_log) > args.tol:
                        raise SystemExit(
                            f"{row['name']}: reconstructed logpsi mismatch "
                            f"{abs(reconstructed_log-exact_log):.3e}"
                        )
                    if "ok_first8" in row and "ok_norm2" in row:
                        gradients = exact_log_gradients_from_fixture(
                            row, theta, site_dims, amplitude
                        )
                        prefix = [complex_pair(value) for value in row["ok_first8"]]
                        for index, expected in enumerate(prefix):
                            if abs(gradients[index] - expected) > 10 * args.tol:
                                raise SystemExit(
                                    f"{row['name']}: reconstructed Ok[{index}] mismatch "
                                    f"{abs(gradients[index]-expected):.3e}"
                                )
                        norm2 = sum(abs(value) ** 2 for value in gradients)
                        if abs(norm2 - float(row["ok_norm2"])) > max(
                            10 * args.tol, 1.0e-8
                        ):
                            raise SystemExit(
                                f"{row['name']}: reconstructed Ok norm mismatch "
                                f"{abs(norm2-float(row['ok_norm2'])):.3e}"
                            )
                    reconstructed_rows += 1
        if "logpsi_env" in row:
            env = complex_pair(row["logpsi_env"])
            max_env_error = max(max_env_error, abs(env - exact))
            if abs(env - exact) > args.tol:
                raise SystemExit(
                    f"{row['name']}: env/exact logpsi mismatch {abs(env-exact):.3e}"
                )
            if "heisenberg_eloc" in row:
                energy = complex_pair(row["heisenberg_eloc"])
                if not math.isfinite(energy.real) or not math.isfinite(energy.imag):
                    raise SystemExit(f"non-finite energy in {row['name']}")
            if "ok_length" in row:
                if int(row["ok_length"]) <= 0:
                    raise SystemExit(f"empty Ok in {row['name']}")
                if float(row["ok_norm2"]) <= 0.0:
                    raise SystemExit(f"non-positive Ok norm in {row['name']}")
        else:
            boundary_errors += 1
            if "logpsi_env_error" not in row:
                raise SystemExit(f"{row['name']}: no env value or error")

    print(
        f"rows={len(rows)} logpsi_rows={len(logpsi_rows)} "
        f"reconstructed_rows={reconstructed_rows} "
        f"boundary_errors={boundary_errors} max_env_error={max_env_error:.3e}"
    )


if __name__ == "__main__":
    main()
