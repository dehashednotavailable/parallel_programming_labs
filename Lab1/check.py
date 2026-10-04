import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from generate import generate_matrices, write_matrix
from run_experiments import run_program
from verify import compare_result, read_matrix, verify_files


def check_case(binary, directory, name, a, b, known_result=None):
    a_path = directory / "A.txt"
    b_path = directory / "B.txt"
    c_path = directory / "C.txt"
    write_matrix(a_path, a)
    write_matrix(b_path, b)
    run_program(binary, a_path, b_path, c_path, a.shape[0])
    accuracy = verify_files(a_path, b_path, c_path)
    if not accuracy["passed"]:
        raise RuntimeError(f"{name}: NumPy verification failed")
    if known_result is not None and not compare_result(known_result, read_matrix(c_path))["passed"]:
        raise RuntimeError(f"{name}: result differs from the known answer")
    print(f"PASS {name}: max_abs={accuracy['accuracy_max_abs']:.6g}")


def expect_program_error(binary, arguments, name):
    result = subprocess.run(
        [str(binary), *map(str, arguments)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        raise RuntimeError(f"{name}: invalid input was accepted")
    print(f"PASS {name}")


def check_invalid_inputs(binary, directory):
    a_path = directory / "A.txt"
    b_path = directory / "B.txt"
    c_path = directory / "C.txt"
    write_matrix(b_path, np.eye(2))
    expect_program_error(binary, [], "missing command-line arguments")
    expect_program_error(binary, [a_path, b_path], "incomplete command-line arguments")
    expect_program_error(binary, [directory / "missing.txt", b_path, c_path], "missing input file")

    invalid = {
        "zero dimension": "0\n",
        "negative dimension": "-2\n1 2\n3 4\n",
        "fractional dimension": "2.5\n1 2\n3 4\n",
        "nonnumeric dimension": "two\n1 2\n3 4\n",
        "too few elements": "2\n1 2\n3\n",
        "extra elements": "2\n1 2\n3 4 5\n",
        "nonnumeric element": "2\n1 2\n3 nope\n",
        "NaN element": "2\n1 2\n3 nan\n",
        "infinite element": "2\n1 2\n3 inf\n",
        "overflowing element": "2\n1 2\n3 1e309\n",
    }
    for name, content in invalid.items():
        a_path.write_text(content, encoding="utf-8")
        expect_program_error(binary, [a_path, b_path, c_path], name)

    write_matrix(a_path, np.eye(3))
    expect_program_error(binary, [a_path, b_path, c_path], "mismatched dimensions")
    write_matrix(a_path, np.eye(2))
    expect_program_error(binary, [a_path, directory / "missing-B.txt", c_path], "missing second input file")
    expect_program_error(binary, [a_path, b_path, directory / "missing-dir" / "C.txt"], "unavailable output file")
    expect_program_error(binary, [a_path, b_path, directory], "directory as output file")


def check_verifier(directory):
    expected = np.eye(2)
    wrong = expected.copy()
    wrong[0, 1] = 1
    if compare_result(expected, wrong)["passed"]:
        raise RuntimeError("Verifier accepted an incorrect result")

    for name, actual in (
        ("wrong result shape", np.eye(3)),
        ("non-finite result", np.full((2, 2), np.nan)),
    ):
        try:
            compare_result(expected, actual)
        except ValueError:
            print(f"PASS verifier rejects {name}")
        else:
            raise RuntimeError(f"Verifier accepted {name}")

    malformed = directory / "malformed.txt"
    malformed.write_text("2\n1 2\n3 4 5\n", encoding="utf-8")
    try:
        read_matrix(malformed)
    except ValueError:
        print("PASS verifier rejects extra elements")
    else:
        raise RuntimeError("Verifier accepted extra matrix elements")
    print("PASS verifier rejects an incorrect result")


def run_checks(binary):
    binary = binary.resolve()
    if not binary.is_file():
        raise ValueError(f"Binary not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="lab1-checks-") as temporary:
        directory = Path(temporary)
        check_case(binary, directory, "1 x 1", np.array([[-2.5]]), np.array([[4.0]]), np.array([[-10.0]]))
        check_case(
            binary, directory, "known noncommutative 2 x 2",
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[19.0, 22.0], [43.0, 50.0]]),
        )
        a, b = generate_matrices(4, 42)
        check_case(binary, directory, "right identity", a, np.eye(4), a)
        check_case(binary, directory, "left identity", np.eye(4), b, b)
        check_case(binary, directory, "zero matrix", a, np.zeros((4, 4)), np.zeros((4, 4)))
        check_case(
            binary, directory, "negative and fractional values",
            np.array([[-1.0, 2.5], [3.25, -4.0]]),
            np.array([[0.5, -2.0], [0.0, 1.75]]),
            np.array([[-0.5, 6.375], [1.625, -13.5]]),
        )
        for size in (3, 5, 11, 17):
            a, b = generate_matrices(size, 42 + size)
            check_case(binary, directory, f"random {size} x {size}", a, b)

        check_invalid_inputs(binary, directory)
        check_verifier(directory)
    print("All checks passed")


def main():
    parser = argparse.ArgumentParser(description="Check multiplication, metrics and error handling")
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_checks(args.binary)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Check failed: {error}\n")


if __name__ == "__main__":
    main()
