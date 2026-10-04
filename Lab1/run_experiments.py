import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile

import numpy as np

from generate import generate_matrices, write_matrix
from verify import ATOL, RTOL, compare_result, read_matrix, reference_product


RAW_FIELDS = [
    "size", "repeat", "time_seconds", "matrix_bytes", "scalar_operations",
    "accuracy_max_abs", "passed",
]
SUMMARY_FIELDS = [
    "size", "repeats", "median_seconds", "min_seconds", "max_seconds",
    "stddev_seconds", "matrix_bytes", "scalar_operations", "accuracy_max_abs",
    "passed",
]


def parse_metrics(stdout, size):
    values = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ("size", "time_seconds", "matrix_bytes", "scalar_operations"):
            if key in values:
                raise ValueError(f"Duplicate program metric: {key}")
            values[key] = value.strip()

    try:
        metrics = {
            "size": int(values["size"]),
            "time_seconds": float(values["time_seconds"]),
            "matrix_bytes": int(values["matrix_bytes"]),
            "scalar_operations": int(values["scalar_operations"]),
        }
    except (KeyError, ValueError) as error:
        raise ValueError("Program did not print valid required metrics") from error

    if metrics["size"] != size:
        raise ValueError("Program reported the wrong matrix size")
    if not math.isfinite(metrics["time_seconds"]) or metrics["time_seconds"] < 0:
        raise ValueError("Program reported an invalid execution time")
    if metrics["matrix_bytes"] != 24 * size * size:
        raise ValueError("Program reported the wrong matrix memory volume")
    if metrics["scalar_operations"] != 2 * size ** 3:
        raise ValueError("Program reported the wrong operation count")
    return metrics


def run_program(binary, a_path, b_path, c_path, size):
    if c_path.exists():
        c_path.unlink()
    result = subprocess.run(
        [str(binary), str(a_path), str(b_path), str(c_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Program exited with {result.returncode}: {details}")
    if not c_path.is_file():
        raise RuntimeError("Program did not create the result matrix file")
    return parse_metrics(result.stdout, size)


def collect_environment(args, binary):
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            cpu = result.stdout.strip()
    elif sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break

    try:
        compiler_result = subprocess.run(
            shlex.split(args.compiler) + ["--version"],
            capture_output=True, text=True, check=False,
        )
        compiler_version = compiler_result.stdout.strip() or compiler_result.stderr.strip()
    except OSError as error:
        compiler_version = f"Unavailable: {error}"

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "compiler": args.compiler,
        "compiler_version": compiler_version,
        "flags": args.flags,
        "python": sys.version,
        "numpy": np.__version__,
        "binary": binary.name,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "sizes": args.sizes,
        "repeats": args.repeats,
        "warmups_per_size": 1,
        "seed": args.seed,
        "matrix_generation": "numpy.default_rng(seed).uniform(-10, 10); A then B",
        "timing_scope": "C++ matrix multiplication only, as reported by the executable",
        "standard_deviation": "population",
        "rtol": RTOL,
        "atol": ATOL,
        "numpy_thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }


def save_plot(path, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [row["size"] for row in summaries]
    medians = [row["median_seconds"] for row in summaries]
    lower = [row["median_seconds"] - row["min_seconds"] for row in summaries]
    upper = [row["max_seconds"] - row["median_seconds"] for row in summaries]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(sizes, medians, yerr=[lower, upper], fmt="o-", capsize=4)
    ax.set_xlabel("Matrix dimension n")
    ax.set_ylabel("Multiplication time, seconds")
    ax.set_title("Sequential matrix multiplication: median and min/max range")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def run_experiments(args):
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"Binary not found: {binary}")
    if args.repeats <= 0 or not args.sizes or any(size <= 0 for size in args.sizes):
        raise ValueError("Sizes and repeat count must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError("Matrix sizes must not be repeated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(args, binary)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    summaries = []
    with tempfile.TemporaryDirectory(prefix="lab1-matrices-") as temporary:
        temporary = Path(temporary)
        a_path = temporary / "A.txt"
        b_path = temporary / "B.txt"
        c_path = temporary / "C.txt"
        with (args.output_dir / "raw.csv").open("w", newline="", encoding="utf-8") as raw_file:
            raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS, lineterminator="\n")
            raw_writer.writeheader()

            for size in args.sizes:
                a, b = generate_matrices(size, args.seed)
                write_matrix(a_path, a)
                write_matrix(b_path, b)
                expected = reference_product(a, b)

                run_program(binary, a_path, b_path, c_path, size)
                warmup = compare_result(expected, read_matrix(c_path))
                if not warmup["passed"]:
                    raise RuntimeError(f"Warmup verification failed at n={size}")

                rows = []
                for repeat in range(1, args.repeats + 1):
                    metrics = run_program(binary, a_path, b_path, c_path, size)
                    accuracy = compare_result(expected, read_matrix(c_path))
                    row = {**metrics, "repeat": repeat, **accuracy}
                    raw_writer.writerow(row)
                    raw_file.flush()
                    if not accuracy["passed"]:
                        raise RuntimeError(f"Verification failed at n={size}, repeat={repeat}")
                    rows.append(row)

                times = [row["time_seconds"] for row in rows]
                summary = {
                    "size": size,
                    "repeats": args.repeats,
                    "median_seconds": statistics.median(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "stddev_seconds": statistics.pstdev(times),
                    "matrix_bytes": 24 * size * size,
                    "scalar_operations": 2 * size ** 3,
                    "accuracy_max_abs": max(row["accuracy_max_abs"] for row in rows),
                    "passed": all(row["passed"] for row in rows),
                }
                summaries.append(summary)
                with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
                    writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(summaries)
                print(
                    f"n={size}: median={summary['median_seconds']:.6g} s, "
                    f"max_abs={summary['accuracy_max_abs']:.6g}, passed=True",
                    flush=True,
                )

    save_plot(args.output_dir / "timing.svg", summaries)
    print(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run verified serial matrix multiplication experiments")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--flags", default="unknown; specify the flags used to build --binary")
    args = parser.parse_args()

    try:
        run_experiments(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Experiment error: {error}\n")


if __name__ == "__main__":
    main()
