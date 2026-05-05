
from triton_ptx.kernels import operator_list
from triton_ptx.helpers import has_ptx_code
from triton_ptx.sandbox import OutputVerifier


def test_kernel(kernel, num_runs: int) -> bool:
    kernel_name = type(kernel).__name__
    verifier = OutputVerifier(num_samples=200)

    if not has_ptx_code(getattr(kernel, "ptx", None)):
        return True

    print(f"Testing {kernel_name}")
    passed = verifier.verify(kernel)

    if not passed:
        failure = verifier.last_report.get("failure", {})
        sample_index = int(failure.get("sample_index", 0)) + 1
        print(f"Kernel {kernel_name} failed on run {sample_index}/{verifier.num_samples}")
        return False

    return True


def run_tests():
    failed_kernels = []

    for operator in operator_list:
        kernel = operator()

        if not test_kernel(kernel, 10):
            failed_kernels.append(type(kernel).__name__)

    return failed_kernels


def print_results(failed_kernels: list[str]) -> None:
    if not failed_kernels:
        print("\nAll kernels passed")
        return

    print("\nFailed kernels:")
    for kernel_name in failed_kernels:
        print(f"- {kernel_name}")



def main() -> None:
    failed_kernels = run_tests()
    print_results(failed_kernels)


if __name__ == "__main__":
    main()
