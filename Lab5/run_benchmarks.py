import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess


RAW_FIELDS = ["view", "partition", "threads", "repeat", "serial_seconds", "parallel_seconds",
              "speedup", "worker_min_seconds", "worker_max_seconds", "load_imbalance", "pixels_match"]
SUMMARY_FIELDS = ["view", "partition", "threads", "repeats", "serial_median_seconds",
                  "parallel_median_seconds", "speedup", "worker_min_seconds", "worker_max_seconds",
                  "load_imbalance", "pixels_match"]


def run_program(binary, view, partition, threads, image_prefix=None):
    command = [str(binary), "--view", str(view), "--partition", partition,
               "--threads", str(threads), "--repeats", "5"]
    if image_prefix is not None:
        command.extend(["--image-prefix", str(image_prefix)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Mandelbrot run failed")
    metrics = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    for name in ("serial_seconds", "parallel_seconds", "worker_min_seconds", "worker_max_seconds"):
        metrics[name] = float(metrics[name])
    metrics["speedup"] = float(metrics["speedup"])
    metrics["load_imbalance"] = float(metrics["load_imbalance"])
    if metrics.get("pixels_match") != "true":
        raise RuntimeError(f"Output images do not match: view={view}, threads={threads}, partition={partition}")
    if int(metrics["view"]) != view or int(metrics["threads"]) != threads or metrics["partition"] != partition:
        raise RuntimeError("Program reported a different configuration than the one requested")
    return metrics


def configurations():
    # The first mapping follows the assignment's contiguous image blocks.
    for threads in range(2, 9):
        yield 1, "blocks", threads
    # View 2 and three threads expose the effect of uneven work per row.
    for threads in (2, 3, 8):
        yield 2, "blocks", threads
    # Static cyclic rows spread the expensive part of the image among workers.
    for view in (1, 2):
        for threads in (1, 2, 3, 4, 5, 6, 7, 8, 16):
            yield view, "cyclic", threads


def save_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for partition in ("blocks", "cyclic"):
        group = sorted((row for row in rows if row["view"] == 1 and row["partition"] == partition),
                       key=lambda row: row["threads"])
        ax.plot([row["threads"] for row in group], [row["speedup"] for row in group],
                "o-", label=partition)
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Speedup over serial implementation")
    ax.set_title("Mandelbrot view 1: row assignment and speedup")
    ax.set_xticks((1, 2, 3, 4, 5, 6, 7, 8, 16))
    ax.axhline(1, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Row partition")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark and verify both Mandelbrot row partitions")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3, help="Independent runs per configuration")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Binary not found: {binary}")
    if args.repeats <= 0:
        parser.error("Repeat count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    compiler = subprocess.run(["c++", "--version"], capture_output=True, text=True, check=False)
    compiler_version = compiler.stdout.splitlines()[0] if compiler.stdout else compiler.stderr.splitlines()[0]
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "compiler": compiler_version,
        "compiler_flags": "-O2 -std=c++17 -Wall -Wextra -Wpedantic -pthread",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "dimensions": [1600, 1200],
        "max_iterations": 256,
        "internal_repeats": 5,
        "external_repeats": args.repeats,
        "verification": "pixel-for-pixel equality with the serial renderer",
        "timing_scope": "render only; image writing is outside the timed region",
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")

    raw_rows = []
    summaries = []
    for view, partition, threads in configurations():
        times = []
        for repeat in range(1, args.repeats + 1):
            prefix = None
            if view == 1 and partition == "cyclic" and threads == 8 and repeat == 1:
                prefix = image_dir / "view1-cyclic-8"
            metrics = run_program(binary, view, partition, threads, prefix)
            row = {"view": view, "partition": partition, "threads": threads, "repeat": repeat,
                   "serial_seconds": metrics["serial_seconds"], "parallel_seconds": metrics["parallel_seconds"],
                   "speedup": metrics["speedup"], "worker_min_seconds": metrics["worker_min_seconds"],
                   "worker_max_seconds": metrics["worker_max_seconds"],
                   "load_imbalance": metrics["load_imbalance"], "pixels_match": True}
            raw_rows.append(row)
            times.append(row)
        serial = statistics.median(row["serial_seconds"] for row in times)
        parallel = statistics.median(row["parallel_seconds"] for row in times)
        summary = {"view": view, "partition": partition, "threads": threads, "repeats": args.repeats,
                   "serial_median_seconds": serial, "parallel_median_seconds": parallel,
                   "speedup": serial / parallel,
                   "worker_min_seconds": statistics.median(row["worker_min_seconds"] for row in times),
                   "worker_max_seconds": statistics.median(row["worker_max_seconds"] for row in times),
                   "load_imbalance": statistics.median(row["load_imbalance"] for row in times),
                   "pixels_match": True}
        print(f"view={view}, partition={partition}, threads={threads}: "
              f"median={parallel:.6g} s, speedup={summary['speedup']:.3f}x, passed=True", flush=True)

        summaries.append(summary)

    for filename, fields, rows in (("raw.csv", RAW_FIELDS, raw_rows), ("summary.csv", SUMMARY_FIELDS, summaries)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    save_plot(args.output_dir / "speedup.svg", summaries)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
