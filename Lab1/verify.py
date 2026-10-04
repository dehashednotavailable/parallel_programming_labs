import argparse
from pathlib import Path

import numpy as np


RTOL = 1e-9
ATOL = 1e-9


def read_matrix(path):
    with Path(path).open("r", encoding="utf-8") as file:
        header = file.readline().strip()
        if not header.isascii() or not header.isdigit() or int(header) <= 0:
            raise ValueError(f"{path}: first line must contain a positive integer")

        size = int(header)
        values = file.read().split()

    if len(values) != size * size:
        raise ValueError(f"{path}: expected {size * size} values, got {len(values)}")

    try:
        matrix = np.asarray(values, dtype=np.float64).reshape(size, size)
    except ValueError as error:
        raise ValueError(f"{path}: invalid matrix value") from error

    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: matrix contains a non-finite value")
    return matrix


def reference_product(a, b):
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be a nonempty square matrix")
    if b.shape != a.shape:
        raise ValueError("A and B must have the same square dimensions")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Input matrices contain a non-finite value")

    try:
        with np.errstate(over="raise", invalid="raise"):
            expected = a @ b
    except FloatingPointError as error:
        raise ValueError("NumPy reference multiplication overflowed") from error

    if not np.isfinite(expected).all():
        raise ValueError("NumPy reference result contains a non-finite value")
    return expected


def compare_result(expected, actual, rtol=RTOL, atol=ATOL):
    if actual.shape != expected.shape:
        raise ValueError(f"Result shape {actual.shape} differs from {expected.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Result or reference contains a non-finite value")

    with np.errstate(over="ignore"):
        max_abs = float(np.max(np.abs(actual - expected)))
    return {
        "passed": bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
        "accuracy_max_abs": max_abs,
    }


def verify_files(a_path, b_path, c_path):
    a = read_matrix(a_path)
    b = read_matrix(b_path)
    actual = read_matrix(c_path)
    return compare_result(reference_product(a, b), actual)


def main():
    parser = argparse.ArgumentParser(description="Verify C against NumPy A @ B")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("c", type=Path)
    args = parser.parse_args()

    try:
        metrics = verify_files(args.a, args.b, args.c)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Verification error: {error}\n")

    print(f"passed={metrics['passed']}")
    print(f"accuracy_max_abs={metrics['accuracy_max_abs']:.17g}")
    print(f"rtol={RTOL}")
    print(f"atol={ATOL}")
    if not metrics["passed"]:
        parser.exit(1, "Verification failed: result differs from NumPy A @ B\n")


if __name__ == "__main__":
    main()
