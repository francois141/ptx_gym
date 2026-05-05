import argparse

from triton_ptx.kernels import operator_list
from triton_ptx.sandbox import OutputVerifier


def test_kernel(kernel, nb_runs: int) -> bool:
    kernel_name = kernel.__class__.__name__
    verifier = OutputVerifier(num_samples=nb_runs)
    print(f"Testing {kernel_name}")

    for run_idx in range(verifier.num_samples):
        inputs = kernel.get_random_input()

        output_triton, _ = kernel.forward_triton(inputs)
        output_torch = kernel.forward_torch(inputs)

        if not verifier.check_similarity(output_triton, output_torch):
            print(f"Kernel {kernel_name} failed on run {run_idx + 1}/{verifier.num_samples}")
            return False

    print(f"Kernel {kernel_name} passed")
    return True


def main(nb_runs: int) -> None:
    failed_kernels = []

    for op in operator_list:
        kernel = op()

        if not test_kernel(kernel, nb_runs):
            failed_kernels.append(kernel.__class__.__name__)

    if failed_kernels:
        print("\nFailed kernels:")
        for name in failed_kernels:
            print(f"- {name}")
    else:
        print("\nAll kernels passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Triton kernels against PyTorch outputs.")
    parser.add_argument(
        "-n",
        "--nb-runs",
        type=int,
        default=250,
        help="Number of random test runs per kernel. Default: 250",
    )

    args = parser.parse_args()
    main(args.nb_runs)
