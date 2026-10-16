import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Lab1"))
from generate import generate_matrices, write_matrix
from verify import verify_files


def run_case(binary, launcher, directory, size, processes, seed):
    a, b = generate_matrices(size, seed)
    a_path, b_path, c_path = directory / "A.txt", directory / "B.txt", directory / "C.txt"
    write_matrix(a_path, a)
    write_matrix(b_path, b)
    c_path.unlink(missing_ok=True)
    result = subprocess.run(
        [*launcher, "-np", str(processes), str(binary), str(a_path), str(b_path), str(c_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"n={size}, processes={processes} failed")

    metrics = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if int(metrics["size"]) != size or int(metrics["processes"]) != processes:
        raise RuntimeError("MPI program reported incorrect dimensions or process count")
    if int(metrics["matrix_bytes"]) != 24 * size * size:
        raise RuntimeError("MPI program reported an incorrect data volume")
    if not verify_files(a_path, b_path, c_path)["passed"]:
        raise RuntimeError(f"NumPy verification failed at n={size}, processes={processes}")
    print(f"PASS n={size}, processes={processes}")


def main():
    parser = argparse.ArgumentParser(description="Check MPI matrix multiplication")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--launcher", default="mpirun --oversubscribe")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Binary not found: {binary}")
    launcher = args.launcher.split()

    with tempfile.TemporaryDirectory(prefix="lab3-check-") as temporary:
        directory = Path(temporary)
        for size, processes in ((1, 1), (1, 4), (2, 4), (17, 2), (33, 4)):
            run_case(binary, launcher, directory, size, processes, 42 + size)

        a_path, b_path, c_path = directory / "A.txt", directory / "B.txt", directory / "C.txt"
        write_matrix(a_path, np.eye(3))
        write_matrix(b_path, np.eye(3))
        c_path.unlink(missing_ok=True)
        result = subprocess.run(
            [*launcher, "-np", "4", str(binary), str(a_path), str(b_path), str(c_path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("MPI rejected uneven row counts")
        if not verify_files(a_path, b_path, c_path)["passed"]:
            raise RuntimeError("MPI uneven-row-count result differs from NumPy")
        print("PASS uneven row distribution")

    print("All MPI checks passed")


if __name__ == "__main__":
    main()
