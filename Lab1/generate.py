import argparse
from pathlib import Path

import numpy as np


def generate_matrices(size, seed=42):
    if size <= 0:
        raise ValueError("Matrix size must be positive")

    rng = np.random.default_rng(seed)
    a = rng.uniform(-10, 10, (size, size))
    b = rng.uniform(-10, 10, (size, size))
    return a, b


def write_matrix(path, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Matrix must be nonempty and contain finite values")

    with Path(path).open("w", encoding="utf-8") as file:
        file.write(f"{matrix.shape[0]}\n")
        np.savetxt(file, matrix, fmt="%.17g")


def main():
    parser = argparse.ArgumentParser(description="Generate two square matrices")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    try:
        a, b = generate_matrices(args.size, args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_matrix(args.output_dir / "A.txt", a)
        write_matrix(args.output_dir / "B.txt", b)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"size={args.size}")
    print(f"seed={args.seed}")
    print(f"A={args.output_dir / 'A.txt'}")
    print(f"B={args.output_dir / 'B.txt'}")


if __name__ == "__main__":
    main()
