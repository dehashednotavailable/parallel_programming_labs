import argparse
from pathlib import Path
import subprocess


def run_check(binary, view, threads, partition):
    result = subprocess.run(
        [str(binary), "--view", str(view), "--threads", str(threads),
         "--partition", partition, "--repeats", "2"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Mandelbrot validation failed")
    metrics = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    expected = {"view": view, "threads": threads, "partition": partition,
                "pixels_match": "true", "width": 1600, "height": 1200}
    for key, value in expected.items():
        if metrics.get(key) != str(value):
            raise RuntimeError(f"Unexpected {key}={metrics.get(key)}; expected {value}")
    print(f"view={view}, threads={threads}, partition={partition}: passed=True", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Check the serial and threaded Mandelbrot images")
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Binary not found: {binary}")

    try:
        for view in (1, 2):
            for partition in ("blocks", "cyclic"):
                for threads in (1, 2, 3, 4, 8, 16):
                    run_check(binary, view, threads, partition)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Check error: {error}\n")


if __name__ == "__main__":
    main()
