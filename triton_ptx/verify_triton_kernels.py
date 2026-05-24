
from triton_ptx.kernels import operator_list
from triton_ptx.sandbox import OutputVerifier


def test_kernel(kernel) -> bool:
    kernel_name = kernel.__class__.__name__
    print(f"Testing {kernel_name}")
    return OutputVerifier().verify_triton_vs_torch(kernel)

def main() -> None:
    failed_kernels = []

    for op in operator_list:
        kernel = op()

        if not test_kernel(kernel):
            failed_kernels.append(kernel.__class__.__name__)

    if failed_kernels:
        print("\nFailed kernels:")
        for name in failed_kernels:
            print(f"- {name}")
    else:
        print("\nAll kernels passed")


if __name__ == "__main__":
    main()
