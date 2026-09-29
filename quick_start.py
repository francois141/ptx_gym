from ptx_gym import dump_kernel_ptx, evaluate_candidate, get_kernel_data, list_kernels

KERNEL_ID = "ReLUFloat16Kernel"


def section(title):
    print(f"\n=== {title} ===")


section("Available kernels")
kernels = list_kernels()
print(f"{len(kernels)} kernels: {', '.join(kernels)}")

# Everything a model needs to write PTX for this kernel. The first call
# autotunes the Triton baseline; later calls reuse the same configuration.
data = get_kernel_data(KERNEL_ID)

section(f"Triton source ({KERNEL_ID})")
print(data["source"])

section("Compilation contract")
system = data["system"]
print(f"target:           {system['target']} (PTX {system['version']}, "
      f"{system['address_size']}-bit)")
print(f"num_warps:        {data['num_warps']} "
      f"({data['num_warps'] * 32} threads per CTA)")
print(f"constexpr values: {data['constexpr_values']}")
print("PTX signature:")
for parameter in data["ptx_signature"]:
    print(f"  {parameter['ptx_type']:<6} {parameter['name']}")

# Submit a candidate. Here it is Triton's own PTX; put your model's PTX here.
candidate_ptx = dump_kernel_ptx(KERNEL_ID)
result = evaluate_candidate(
    KERNEL_ID,
    {"ptx": candidate_ptx, "num_threads_x": data["num_warps"] * 32},
)

section("Evaluation result")
print(f"passed:  {result['passed']}")
print(f"message: {result['message']}")
if result["passed"]:
    print(f"latency: {result['p50']:.4f} ms candidate vs "
          f"{result['triton_p50']:.4f} ms Triton (p50)")
    print(f"speedup: {result['speedup_vs_triton']:.3f}x")
