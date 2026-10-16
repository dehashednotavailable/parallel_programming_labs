#!/usr/bin/env python3
"""Run MPI or CUDA matrix experiments without third-party packages on the cluster."""

import argparse
import csv
from datetime import datetime, timezone
import filecmp
import hashlib
import gzip
import json
import math
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import tempfile


RAW_FIELDS = [
    "size", "processes", "tile_size", "repeat", "time_seconds",
    "end_to_end_seconds", "matrix_bytes", "scalar_operations", "matches_baseline",
]
SUMMARY_FIELDS = [
    "size", "processes", "tile_size", "repeats", "median_seconds",
    "end_to_end_median_seconds", "min_seconds", "max_seconds",
    "stddev_seconds", "speedup", "matches_baseline",
]


def write_input(path, size, seed):
    state = seed
    with path.open("w", encoding="ascii") as output:
        output.write(f"{size}\n")
        for _ in range(size):
            row = []
            for _ in range(size):
                state = (state * 48271) % 2147483647
                row.append(str(state % 19 - 9))
            output.write(" ".join(row) + "\n")


def write_inputs(directory, size, seed):
    a_path = directory / "A.txt"
    b_path = directory / "B.txt"
    write_input(a_path, size, seed + size * 1009)
    write_input(b_path, size, seed + size * 1009 + 1)
    return a_path, b_path


def read_metrics(stdout):
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def run_one(args, binary, a_path, b_path, c_path, size, configuration):
    if args.program == "mpi":
        launcher = shlex.split(args.launcher)
        if Path(launcher[0]).name == "srun":
            command = [*launcher, "--ntasks", str(configuration), str(binary), str(a_path), str(b_path), str(c_path)]
        else:
            command = [*launcher, "-np", str(configuration), str(binary), str(a_path), str(b_path), str(c_path)]
    else:
        command = [str(binary), str(a_path), str(b_path), str(c_path), str(configuration)]

    if c_path.exists():
        c_path.unlink()
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "The matrix program failed")
    if not c_path.is_file():
        raise RuntimeError("The matrix program did not write its result")

    metrics = read_metrics(result.stdout)
    expected = {
        "size": size,
        "matrix_bytes": 24 * size * size,
        "scalar_operations": 2 * size ** 3,
    }
    if args.program == "mpi":
        expected["processes"] = configuration
        time_seconds = float(metrics["time_seconds"])
        end_to_end_seconds = ""
        process_count = configuration
        tile_size = ""
    else:
        expected["tile_size"] = configuration
        time_seconds = float(metrics["time_seconds"])
        end_to_end_seconds = float(metrics["end_to_end_seconds"])
        process_count = ""
        tile_size = configuration
        args.device = metrics["device"]

    for key, value in expected.items():
        if int(metrics[key]) != value:
            raise RuntimeError(f"Program reported {key}={metrics.get(key)}, expected {value}")
    if not math.isfinite(time_seconds) or time_seconds <= 0:
        raise RuntimeError("Program reported an invalid measured time")
    if end_to_end_seconds != "" and (not math.isfinite(end_to_end_seconds) or end_to_end_seconds <= 0):
        raise RuntimeError("Program reported an invalid end-to-end time")

    return {
        "processes": process_count,
        "tile_size": tile_size,
        "time_seconds": time_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "matrix_bytes": 24 * size * size,
        "scalar_operations": 2 * size ** 3,
    }


def save_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(args):
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"Binary not found: {binary}")
    baseline = 1 if args.program == "mpi" else 4
    if not args.configurations or args.configurations[0] != baseline:
        raise ValueError(f"The first configuration must be the {baseline} baseline")
    if len(set(args.sizes)) != len(args.sizes) or any(size <= 0 for size in args.sizes):
        raise ValueError("Sizes must be positive and unique")
    if len(set(args.configurations)) != len(args.configurations):
        raise ValueError("Configurations must be unique")
    if args.repeats <= 0:
        raise ValueError("Repeat count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = args.output_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    args.device = ""
    launcher_version = ""
    if args.program == "mpi":
        version = subprocess.run(
            [*shlex.split(args.launcher), "--version"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, check=False,
        )
        launcher_version = version.stdout.strip() or version.stderr.strip()

    raw_rows = []
    summaries = []
    with tempfile.TemporaryDirectory(prefix="cluster-matrix-") as temporary:
        directory = Path(temporary)
        a_path, b_path, c_path = directory / "A.txt", directory / "B.txt", directory / "C.txt"

        for size in args.sizes:
            write_inputs(directory, size, args.seed)
            reference_path = reference_dir / f"C_{size}.txt.gz"
            baseline_path = directory / f"C_reference_{size}.txt"
            if reference_path.exists():
                reference_path.unlink()
            times_by_configuration = {}
            end_to_end_by_configuration = {}

            for configuration in args.configurations:
                first = run_one(args, binary, a_path, b_path, c_path, size, configuration)
                if configuration == baseline:
                    shutil.copyfile(c_path, baseline_path)
                if not filecmp.cmp(c_path, baseline_path, shallow=False):
                    raise RuntimeError(
                        f"Output differs from baseline at n={size}, configuration={configuration}"
                    )

                times = []
                end_to_end_times = []
                for repeat in range(1, args.repeats + 1):
                    measured = run_one(args, binary, a_path, b_path, c_path, size, configuration)
                    if not filecmp.cmp(c_path, baseline_path, shallow=False):
                        raise RuntimeError(
                            f"Output differs from baseline at n={size}, configuration={configuration}, repeat={repeat}"
                        )
                    raw_rows.append({
                        "size": size,
                        "processes": measured["processes"],
                        "tile_size": measured["tile_size"],
                        "repeat": repeat,
                        "time_seconds": measured["time_seconds"],
                        "end_to_end_seconds": measured["end_to_end_seconds"],
                        "matrix_bytes": measured["matrix_bytes"],
                        "scalar_operations": measured["scalar_operations"],
                        "matches_baseline": True,
                    })
                    times.append(measured["time_seconds"])
                    if measured["end_to_end_seconds"] != "":
                        end_to_end_times.append(measured["end_to_end_seconds"])

                times_by_configuration[configuration] = times
                end_to_end_by_configuration[configuration] = end_to_end_times
                print(
                    f"n={size}, configuration={configuration}: "
                    f"median={statistics.median(times):.6g}s, matches_baseline=True",
                    flush=True,
                )

            with baseline_path.open("rb") as source, gzip.open(str(reference_path), "wb") as destination:
                shutil.copyfileobj(source, destination)

            base_median = statistics.median(times_by_configuration[baseline])
            for configuration in args.configurations:
                times = times_by_configuration[configuration]
                end_to_end_times = end_to_end_by_configuration[configuration]
                summaries.append({
                    "size": size,
                    "processes": configuration if args.program == "mpi" else "",
                    "tile_size": configuration if args.program == "cuda" else "",
                    "repeats": args.repeats,
                    "median_seconds": statistics.median(times),
                    "end_to_end_median_seconds": statistics.median(end_to_end_times) if end_to_end_times else "",
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "stddev_seconds": statistics.pstdev(times),
                    "speedup": base_median / statistics.median(times),
                    "matches_baseline": True,
                })

    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "program": args.program,
        "host": platform.node(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "python": platform.python_version(),
        "compiler": args.compiler,
        "compiler_flags": args.flags,
        "binary": binary.name,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "mpi_launcher": args.launcher if args.program == "mpi" else "",
        "launcher_version": launcher_version,
        "mpi_version": args.mpi_version if args.program == "mpi" else "",
        "gpu_device": args.device,
        "sizes": args.sizes,
        "configurations": args.configurations,
        "repeats": args.repeats,
        "warmups_per_configuration": 1,
        "seed": args.seed,
        "input_generator": "Park-Miller LCG, integer matrix values in [-9, 9]",
        "reference_outputs": "results/reference/C_<size>.txt.gz; local NumPy verification required",
        "parallel_consistency": "Each run output compared byte-for-byte with the baseline configuration",
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID", ""),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_csv(args.output_dir / "raw.csv", RAW_FIELDS, raw_rows)
    save_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    print(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", choices=("mpi", "cuda"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    parser.add_argument("--configurations", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--mpi-version", default="")
    parser.add_argument("--compiler", default="mpicxx or nvcc")
    parser.add_argument("--flags", default="-O2; see Slurm job script")
    args = parser.parse_args()
    try:
        run_experiments(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Experiment error: {error}\n")


if __name__ == "__main__":
    main()
