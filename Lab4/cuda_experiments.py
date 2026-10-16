import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Lab1"))
from generate import generate_matrices, write_matrix
from verify import compare_result, read_matrix, reference_product


def run_one(binary, a_path, b_path, c_path, size, tile):
    c_path.unlink(missing_ok=True)
    result = subprocess.run([str(binary), str(a_path), str(b_path), str(c_path), str(tile)], capture_output=True, text=True)
    if result.returncode or not c_path.is_file():
        raise RuntimeError(result.stderr.strip() or "CUDA program failed to create C")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    expected = {"size": size, "tile_size": tile, "matrix_bytes": 24 * size * size, "scalar_operations": 2 * size**3}
    if any(int(fields[key]) != value for key, value in expected.items()):
        raise RuntimeError("CUDA program reported inconsistent experiment metrics")
    elapsed, wall = float(fields["time_seconds"]), float(fields["end_to_end_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(wall) or wall <= 0:
        raise RuntimeError("CUDA program reported an invalid time")
    return fields["device"], elapsed, wall


def plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for tile in sorted({row["tile_size"] for row in rows}):
        group = [row for row in rows if row["tile_size"] == tile]
        ax.plot([r["size"] for r in group], [r["median_seconds"] for r in group], "o-", label=f"{tile} x {tile}")
    ax.set_xlabel("Matrix dimension n")
    ax.set_ylabel("Kernel time, seconds")
    ax.set_title("CUDA matrix multiplication: block size comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(title="CUDA block")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def run(args):
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"Binary not found: {binary}")
    if not args.sizes or any(n <= 0 for n in args.sizes) or args.repeats <= 0:
        raise ValueError("Sizes and repeat count must be positive")
    if not args.tiles or 4 not in args.tiles or any(tile not in (4, 8, 16, 32) for tile in args.tiles):
        raise ValueError("Select tile sizes 4, 8, 16, 32 and include the 4 x 4 baseline")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.tiles)) != len(args.tiles):
        raise ValueError("Repeated sizes and tile configurations are not allowed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiler = subprocess.run([args.compiler, "--version"], capture_output=True, text=True, check=False)
    env = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "os": platform.platform(),
           "architecture": platform.machine(), "compiler": args.compiler,
           "compiler_version": compiler.stdout.strip() or compiler.stderr.strip(), "compiler_flags": args.flags,
           "binary": binary.name, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
           "sizes": args.sizes, "tile_sizes": args.tiles, "repeats": args.repeats,
           "warmups": 1, "seed": args.seed, "verification": "NumPy A @ B, rtol=1e-9, atol=1e-9",
           "timing_scope": "CUDA events for kernel; wall time includes allocation, copies and file output"}
    raw, summary = [], []
    device = "unknown"
    with tempfile.TemporaryDirectory(prefix="lab4-matrices-") as temporary:
        directory = Path(temporary)
        a_path, b_path, c_path = directory / "A.txt", directory / "B.txt", directory / "C.txt"
        for size in args.sizes:
            a, b = generate_matrices(size, args.seed)
            write_matrix(a_path, a)
            write_matrix(b_path, b)
            expected = reference_product(a, b)
            for tile in args.tiles:
                device, _, _ = run_one(binary, a_path, b_path, c_path, size, tile)
                if not compare_result(expected, read_matrix(c_path))["passed"]:
                    raise RuntimeError(f"Warmup verification failed at n={size}, tile={tile}")
                times, walls, errors = [], [], []
                for repeat in range(1, args.repeats + 1):
                    device, elapsed, wall = run_one(binary, a_path, b_path, c_path, size, tile)
                    accuracy = compare_result(expected, read_matrix(c_path))
                    if not accuracy["passed"]:
                        raise RuntimeError(f"NumPy verification failed at n={size}, tile={tile}")
                    raw.append({"size": size, "tile_size": tile, "repeat": repeat, "time_seconds": elapsed,
                                "end_to_end_seconds": wall, "matrix_bytes": 24 * size * size,
                                "scalar_operations": 2 * size**3, **accuracy})
                    times.append(elapsed)
                    walls.append(wall)
                    errors.append(accuracy["accuracy_max_abs"])
                summary.append({"size": size, "tile_size": tile, "repeats": args.repeats,
                                "median_seconds": statistics.median(times), "min_seconds": min(times),
                                "max_seconds": max(times), "stddev_seconds": statistics.pstdev(times),
                                "end_to_end_median_seconds": statistics.median(walls),
                                "accuracy_max_abs": max(errors), "passed": True})
                print(f"n={size}, tile={tile}: median={statistics.median(times):.6g}s, GPU={device}, passed=True", flush=True)

    baseline = {row["size"]: row["median_seconds"] for row in summary if row["tile_size"] == 4}
    for row in summary:
        row["speedup_vs_tile_4"] = baseline[row["size"]] / row["median_seconds"]
    env["gpu_device"] = device
    fields = {"raw.csv": list(raw[0].keys()), "summary.csv": list(summary[0].keys())}
    for filename, rows in (("raw.csv", raw), ("summary.csv", summary)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields[filename], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(args.output_dir / "blocks.svg", summary)
    print(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark and verify CUDA matrix multiplication")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    parser.add_argument("--tiles", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--compiler", default="nvcc")
    parser.add_argument("--flags", default="-O2 -arch=sm_20")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Experiment error: {error}\n")


if __name__ == "__main__":
    main()
