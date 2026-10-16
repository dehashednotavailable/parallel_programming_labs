import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Lab1"))
from generate import generate_matrices, write_matrix
from verify import ATOL, RTOL, compare_result, read_matrix, reference_product


RAW_FIELDS = ["size", "processes", "repeat", "time_seconds", "matrix_bytes", "scalar_operations", "accuracy_max_abs", "passed"]
SUMMARY_FIELDS = ["size", "processes", "repeats", "median_seconds", "min_seconds", "max_seconds", "stddev_seconds", "speedup", "accuracy_max_abs", "passed"]


def run_program(binary, launcher, a_path, b_path, c_path, size, processes):
    c_path.unlink(missing_ok=True)
    result = subprocess.run(
        [*launcher, "-np", str(processes), str(binary), str(a_path), str(b_path), str(c_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "MPI program failed")
    if not c_path.is_file():
        raise RuntimeError("MPI program did not save the result matrix")

    metrics = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    expected = {"size": size, "processes": processes, "matrix_bytes": 24 * size * size,
                "scalar_operations": 2 * size ** 3}
    for key, value in expected.items():
        if int(metrics[key]) != value:
            raise RuntimeError(f"MPI program reported {key}={metrics.get(key)}, expected {value}")
    elapsed = float(metrics["time_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("MPI program reported an invalid elapsed time")
    return elapsed


def save_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for processes in sorted({row["processes"] for row in rows}):
        group = [row for row in rows if row["processes"] == processes]
        ax.plot([row["size"] for row in group], [row["speedup"] for row in group], "o-", label=f"{processes} processes")
    ax.set_xlabel("Matrix dimension n")
    ax.set_ylabel("Speedup over 1 process")
    ax.set_title("MPI matrix multiplication: speedup by matrix size")
    ax.axhline(1, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="svg")
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def run_experiments(args):
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"Binary not found: {binary}")
    if not args.processes or 1 not in args.processes or any(value <= 0 for value in args.processes):
        raise ValueError("Process counts must be positive and include 1")
    if len(set(args.processes)) != len(args.processes) or len(set(args.sizes)) != len(args.sizes):
        raise ValueError("Process counts and sizes must not be repeated")
    if args.repeats <= 0 or any(value <= 0 for value in args.sizes):
        raise ValueError("Sizes and repeat count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher = shlex.split(args.launcher)
    mpi_version = subprocess.run([launcher[0], "--version"], capture_output=True, text=True, check=False)
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "mpi_launcher": args.launcher,
        "mpi_version": mpi_version.stdout.splitlines()[0] if mpi_version.stdout else mpi_version.stderr.splitlines()[0],
        "compiler": args.compiler,
        "compiler_flags": args.flags,
        "binary": binary.name,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "sizes": args.sizes,
        "process_counts": args.processes,
        "repeats": args.repeats,
        "warmups_per_configuration": 1,
        "seed": args.seed,
        "verification": "NumPy A @ B, rtol=1e-9, atol=1e-9",
        "timing_scope": "MPI wall time including matrix distribution, multiplication and result collection",
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raw_rows = []
    summaries = []
    with tempfile.TemporaryDirectory(prefix="lab3-matrices-") as temporary:
        directory = Path(temporary)
        a_path, b_path, c_path = directory / "A.txt", directory / "B.txt", directory / "C.txt"
        for size in args.sizes:
            a, b = generate_matrices(size, args.seed)
            write_matrix(a_path, a)
            write_matrix(b_path, b)
            expected = reference_product(a, b)

            for processes in args.processes:
                run_program(binary, launcher, a_path, b_path, c_path, size, processes)
                accuracy = compare_result(expected, read_matrix(c_path))
                if not accuracy["passed"]:
                    raise RuntimeError(f"Warmup result failed NumPy verification at n={size}, processes={processes}")

                times = []
                errors = []
                for repeat in range(1, args.repeats + 1):
                    elapsed = run_program(binary, launcher, a_path, b_path, c_path, size, processes)
                    accuracy = compare_result(expected, read_matrix(c_path))
                    if not accuracy["passed"]:
                        raise RuntimeError(f"NumPy verification failed at n={size}, processes={processes}, repeat={repeat}")
                    raw_rows.append({"size": size, "processes": processes, "repeat": repeat,
                                     "time_seconds": elapsed, "matrix_bytes": 24 * size * size,
                                     "scalar_operations": 2 * size ** 3,
                                     "accuracy_max_abs": accuracy["accuracy_max_abs"], "passed": accuracy["passed"]})
                    times.append(elapsed)
                    errors.append(accuracy["accuracy_max_abs"])

                summary = {"size": size, "processes": processes, "repeats": args.repeats,
                           "median_seconds": statistics.median(times), "min_seconds": min(times),
                           "max_seconds": max(times), "stddev_seconds": statistics.pstdev(times),
                           "accuracy_max_abs": max(errors), "passed": True}
                summaries.append(summary)
                print(f"n={size}, processes={processes}: median={summary['median_seconds']:.6g} s, passed=True", flush=True)

    serial = {row["size"]: row["median_seconds"] for row in summaries if row["processes"] == 1}
    for summary in summaries:
        summary["speedup"] = serial[summary["size"]] / summary["median_seconds"]

    for filename, fields, rows in (("raw.csv", RAW_FIELDS, raw_rows), ("summary.csv", SUMMARY_FIELDS, summaries)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    save_plot(args.output_dir / "speedup.svg", summaries)
    print(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark and verify MPI matrix multiplication")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--launcher", default="mpirun --oversubscribe")
    parser.add_argument("--compiler", default="mpicxx")
    parser.add_argument("--flags", default="-O2 -std=c++17 -Wall -Wextra -Wpedantic")
    args = parser.parse_args()
    try:
        run_experiments(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Experiment error: {error}\n")


if __name__ == "__main__":
    main()
