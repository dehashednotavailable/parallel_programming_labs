#!/usr/bin/env python3
"""Verify downloaded cluster reference matrices with NumPy and plot the results."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Lab1"))
from verify import compare_result, reference_product


def make_matrix(size, seed, generator_name):
    if "random.Random" in generator_name:
        generator = random.Random(seed)
        return np.fromiter(
            (int(generator.random() * 19) - 9 for _ in range(size * size)),
            dtype=np.float64,
            count=size * size,
        ).reshape(size, size)

    state = seed
    values = np.empty(size * size, dtype=np.float64)
    for index in range(size * size):
        state = (state * 48271) % 2147483647
        values[index] = state % 19 - 9
    return values.reshape(size, size)


def read_reference(path, expected_size):
    with gzip.open(str(path), "rt", encoding="ascii") as source:
        header = source.readline().strip()
        if not header.isdigit() or int(header) != expected_size:
            raise ValueError(f"{path}: invalid matrix size header")
        values = np.fromstring(source.read(), dtype=np.float64, sep=" ")
    if values.size != expected_size * expected_size or not np.isfinite(values).all():
        raise ValueError(f"{path}: invalid matrix contents")
    return values.reshape(expected_size, expected_size)


def save_plot(path, program, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if program == "mpi":
        for processes in sorted({int(row["processes"]) for row in rows}):
            group = [row for row in rows if int(row["processes"]) == processes]
            ax.plot([int(row["size"]) for row in group], [float(row["speedup"]) for row in group],
                    "o-", label=f"{processes} processes")
        ax.set_xlabel("Matrix dimension n")
        ax.set_ylabel("Speedup over one process")
        ax.set_title("MPI matrix multiplication on Sergey Korolev")
        ax.axhline(1, color="gray", linewidth=0.8)
        ax.legend()
        filename = "speedup"
    else:
        for tile in sorted({int(row["tile_size"]) for row in rows}):
            group = [row for row in rows if int(row["tile_size"]) == tile]
            ax.plot([int(row["size"]) for row in group], [float(row["median_seconds"]) for row in group],
                    "o-", label=f"{tile} x {tile}")
        ax.set_xlabel("Matrix dimension n")
        ax.set_ylabel("Kernel time, seconds")
        ax.set_title("CUDA matrix multiplication on Sergey Korolev")
        ax.legend(title="CUDA block")
        filename = "blocks"

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    svg_path = path / f"{filename}.svg"
    fig.savefig(svg_path, format="svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    fig.savefig(path / f"{filename}.png", dpi=160)
    plt.close(fig)


def verify(args):
    results_dir = args.results_dir.resolve()
    environment = json.loads((results_dir / "environment.json").read_text(encoding="utf-8"))
    with (results_dir / "summary.csv").open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    seed = int(environment["seed"])
    generator_name = environment.get("input_generator", "Park-Miller LCG")
    verification_rows = []
    for size in environment["sizes"]:
        a = make_matrix(size, seed + size * 1009, generator_name)
        b = make_matrix(size, seed + size * 1009 + 1, generator_name)
        actual = read_reference(results_dir / "reference" / f"C_{size}.txt.gz", size)
        metrics = compare_result(reference_product(a, b), actual)
        verification_rows.append({
            "size": size,
            "accuracy_max_abs": metrics["accuracy_max_abs"],
            "passed": metrics["passed"],
        })

    verification_path = results_dir / "verification.csv"
    with verification_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["size", "accuracy_max_abs", "passed"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(verification_rows)

    save_plot(results_dir, args.program, rows)
    for result in verification_rows:
        print(f"n={result['size']}, NumPy passed={result['passed']}, max_abs={result['accuracy_max_abs']:.6g}")
    if not all(row["passed"] for row in verification_rows):
        raise ValueError("At least one cluster result differs from NumPy A @ B")
    if any(row["matches_baseline"].strip().lower() not in ("true", "1") for row in rows):
        raise ValueError("At least one configuration did not match the baseline output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", choices=("mpi", "cuda"), required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Verification error: {error}\n")


if __name__ == "__main__":
    main()
