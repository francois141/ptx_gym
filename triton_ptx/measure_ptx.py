#!/usr/bin/env python3
from __future__ import annotations

from triton_ptx.sandbox import PTXBenchmarkRunner
from triton_ptx.kernels import kernel_list

def main():
    runner = PTXBenchmarkRunner(kernel_list)
    print(runner.run())


if __name__ == "__main__":
    main()
