#!/usr/bin/env python3
import argparse
from pathlib import Path


from triton_ptx.helpers.environment import is_gpu_available
from triton_ptx.helpers.triton import dump_kernel_ptx
from triton_ptx.kernels import kernel_list


def dump_and_save_ptx_kernel(kernel, output_dir: Path) -> Path:

    kernel_name = kernel.__class__.__name__
    ptx = dump_kernel_ptx(kernel)

    if not ptx:
        raise RuntimeError(f"Triton did not produce PTX for {kernel_name}")

    output_path = output_dir / f"{kernel_name}.ptx"
    output_path.write_text(ptx)

    return output_path


def main(output_dir: str) -> None:
    if not is_gpu_available():
        raise RuntimeError("CUDA is required to compile and dump Triton PTX.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_kernels = []

    for op in kernel_list:
        kernel = op()
        kernel_name = kernel.__class__.__name__

        print(f"Compiling {kernel_name}")

        try:
            output_path = dump_and_save_ptx_kernel(kernel, output_dir)
            print(f"Dumped PTX to {output_path}")
        except Exception as exc:
            print(f"Failed to dump PTX for {kernel_name}: {exc}")
            failed_kernels.append(kernel_name)

    if failed_kernels:
        print("\nFailed kernels:")
        for name in failed_kernels:
            print(f"- {name}")
    else:
        print("\nAll PTX files generated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile Triton operators and dump their generated PTX."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="triton_generated_ptx",
        help="Directory where generated PTX files will be saved.",
    )

    args = parser.parse_args()
    main(args.output_dir)
