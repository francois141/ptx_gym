import torch
import triton
import triton.language as tl


generated_kernel = """.version 8.8
.target sm_120a
.address_size 64

.visible .entry add_kernel(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 output_ptr,
    .param .u32 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
) {
    .reg .pred      %p<3>;
    .reg .b32       %r<10>;
    .reg .f32       %f<5>;
    .reg .b64       %rd<15>;

    // 1. Get IDs
    mov.u32         %r0, %ctaid.x;
    mov.u32         %r1, %tid.x;
    
    // 2. Load Params
    ld.param.u64    %rd1, [x_ptr];
    ld.param.u64    %rd2, [y_ptr];
    ld.param.u64    %rd3, [output_ptr];
    ld.param.u32    %r2, [n_elements];

    // 3. Calculate index: (ctaid.x * 1024) + tid.x
    // We use .u32 for the math then convert to .u64 for the pointer offset
    mul.lo.u32      %r3, %r0, 1024;
    add.u32         %r4, %r3, %r1;
    
    // 4. Boundary Check (Fixing the setp error)
    // Compare .u32 index against .u32 n_elements
    setp.ge.u32     %p1, %r4, %r2;
    @%p1 bra        DONE;

    // 5. Convert index to 64-bit for memory math
    cvt.u64.u32     %rd4, %r4;
    mul.wide.u32    %rd5, %r4, 4; // Correct: .u32 * immediate -> .u64 offset

    // 6. Address calculation
    add.u64         %rd6, %rd1, %rd5; 
    add.u64         %rd7, %rd2, %rd5;
    add.u64         %rd8, %rd3, %rd5;

    // 7. Load, Add, Store
    ld.global.f32   %f1, [%rd6];
    ld.global.f32   %f2, [%rd7];
    add.f32         %f3, %f1, %f2;
    st.global.f32   [%rd8], %f3;

DONE:
    ret;
}"""

# --- 1. THE TRITON KERNEL ---
@triton.jit(ptx=generated_kernel)
def add_kernel(
    x_ptr,           # Pointer to first input vector
    y_ptr,           # Pointer to second input vector
    output_ptr,      # Pointer to output vector
    n_elements,      # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program handles
):
    # Determine which block of data this program instance is responsible for
    pid = tl.program_id(0)
    
    # Map the program ID to the memory offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to avoid memory access violations at the end of the vector
    mask = offsets < n_elements

    # Load data from DRAM to SRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the actual math
    output = x + y

    # Store the result back from SRAM to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

# --- 2. THE PYTORCH WRAPPER ---
def triton_add(x: torch.Tensor, y: torch.Tensor):
    # Ensure inputs are contiguous and on the same CUDA device
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # The grid function defines how many kernels to launch in parallel.
    # It takes a metadata dictionary (containing BLOCK_SIZE) as input.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Execute the kernel
    add_kernel[grid](
        x, y, output, 
        n_elements, 
        BLOCK_SIZE=1024
    )
    
    return output

# --- 3. VERIFICATION AND BENCHMARKING ---
if __name__ == "__main__":
    # Initialize data
    size = 16
    print(f"Running Vector Addition on {size} elements...")
    
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    # Run Triton version
    output_triton = triton_add(x, y)

    # Run standard PyTorch version (Baseline)
    output_torch = x + y

    # Verify accuracy
    if torch.allclose(output_torch, output_triton):
        print("✅ Success: Triton results match PyTorch!")
    else:
        print("❌ Failure: Results do not match.")

    print("Triton Output (first 10 elements):", output_triton[:])
    print("PyTorch Output (first 10 elements):", output_torch[:])

    # Simple timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(1): triton_add(x, y)
    end_event.record()
    
    torch.cuda.synchronize()
    print(f"Average Triton Time: {start_event.elapsed_time(end_event) / 100:.3f} ms")