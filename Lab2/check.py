import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Lab1"))
from generate import generate_matrices, write_matrix
from verify import verify_files


def run_case(binary, directory, size, threads, seed):
    a, b = generate_matrices(size, seed)
    a_path = directory / "A.txt"
    b_path = directory / "B.txt"
    c_path = directory / "C.txt"
    write_matrix(a_path, a)
    write_matrix(b_path, b)
    result = subprocess.run(
        [str(binary), str(a_path), str(b_path), str(c_path), str(threads)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"n={size}, threads={threads} failed")

    metrics = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if int(metrics["size"]) != size or int(metrics["threads_requested"]) != threads:
        raise RuntimeError("The program reported incorrect dimensions or requested threads")
    if int(metrics["threads_used"]) != threads:
        raise RuntimeError("The OpenMP runtime did not start the requested team size")
    if int(metrics["matrix_bytes"]) != 24 * size * size:
        raise RuntimeError("The program reported an incorrect data volume")
    if not verify_files(a_path, b_path, c_path)["passed"]:
        raise RuntimeError(f"NumPy verification failed at n={size}, threads={threads}")

    print(f"PASS n={size}, threads={threads}")


def main():
    parser = argparse.ArgumentParser(description="Check OpenMP matrix multiplication")
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Binary not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="lab2-check-") as temporary:
        directory = Path(temporary)
        for size, threads in ((1, 1), (2, 2), (7, 4), (17, 8), (33, 10)):
            run_case(binary, directory, size, threads, 42 + size)

        a_path = directory / "A.txt"
        b_path = directory / "B.txt"
        c_path = directory / "C.txt"
        write_matrix(a_path, np.eye(2))
        write_matrix(b_path, np.eye(2))
        for threads in ("0", "-1", "two", "2x"):
            result = subprocess.run(
                [str(binary), str(a_path), str(b_path), str(c_path), threads],
                capture_output=True, text=True, check=False,
            )
            if result.returncode == 0:
                raise RuntimeError(f"Invalid thread count was accepted: {threads}")
        print("PASS invalid thread-count checks")
    print("All OpenMP checks passed")


if __name__ == "__main__":
    main()
