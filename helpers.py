import inspect
import triton.language as tl
import torch
import time # Added for performance measurement

import inspect
import torch
import triton
import triton.language as tl

try:
    from IPython.display import display as ipython_display
except ImportError:
    ipython_display = None


def display(obj):
    if ipython_display is not None:
        ipython_display(obj)
        return

    if hasattr(obj, "to_string"):
        print(obj.to_string(index=False))
    else:
        print(obj)

def extract_ptx(compiled_kernel, *, print_ttir=False, print_ttgir=False, print_llir=False):

    if print_ttir:
        print(f"ttir: {compiled_kernel.asm.get('ttir', None)[:2000]}")
        print("-"*100 + "\n")

    if print_ttgir:
        print(f"ttgir: {compiled_kernel.asm.get('ttgir', None)[:1000]}")
        print("-"*100 + "\n")

    if print_llir:
        print(f"llir: {compiled_kernel.asm.get('llir', None)[:1000]}")
        print("-"*100 + "\n")

    return compiled_kernel.asm.get('ptx', None)

def jit_fixed_parameters(fn=None, **triton_kwargs):
    """
    A decorator that automatically populates 'do_not_specialize' for any
    parameter NOT marked as tl.constexpr, then calls triton.jit.
    This implies that the ptx signature will keep this argument and remove 
    only tl.constexpr arguments, which are fixed at compile time and not parametric in the ptx code.
    """
    def decorator(func):
        params = inspect.signature(func).parameters
        do_not_specialize_list = []
        
        for name, param in params.items():
            if param.annotation is not tl.constexpr:
                do_not_specialize_list.append(name)

        return triton.jit(fn=func, do_not_specialize=do_not_specialize_list, **triton_kwargs)

    if fn is None:
        return decorator

    return decorator(fn)

def print_add_kernel_source():
    source = inspect.getsource(TritonOperator.kernel)
    print("\n--- Source Code ---")
    print(source)

    print("\n--- Argument List ---")
    signature = inspect.signature(TritonOperator.kernel)
    for name, param in signature.parameters.items():
        annotation_str = f", Annotation: {param.annotation.__qualname__}" if param.annotation is not inspect.Parameter.empty else ""
        print(f"Argument: {name}, Default: {param.default if param.default is not inspect.Parameter.empty else 'None'}, Kind: {param.kind}{annotation_str}")

def check_similarity(output1, output2, atol=3e-4, rtol=1e-3):
    is_close = torch.allclose(output1, output2, atol=atol, rtol=rtol)

    if not is_close:
        # Calculate the absolute difference element-wise
        diff = torch.abs(output1 - output2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        print(f"Outputs are not similar: {is_close}")
        print(f"Max Difference: {max_diff:.3e}")
        print(f"Mean Difference: {mean_diff:.3e}")

    return is_close

def is_valid_operator(operator_obj):
    """Checks if an object conforms to the operator interface."""
    required_methods = [
        '__init__',
        'kernel',
        'get_random_input',
        'forward_triton',
        'forward_torch'
    ]

    # Check for presence and callability of each method
    for method_name in required_methods:
        if not hasattr(operator_obj, method_name):
            print(f"Error: Operator is missing method: {method_name}")
            return False
        if not callable(getattr(operator_obj, method_name)):
            print(f"Error: Method {method_name} in operator is not callable.")
            return False

    forward_triton_sig = inspect.signature(getattr(operator_obj, "forward_triton"))
    if "ptx" not in forward_triton_sig.parameters:
        print("Error: 'forward_triton' must accept a 'ptx' argument.")
        return False

    # Special check for kernel to be a static method (or at least callable directly from class)
    if not inspect.isfunction(getattr(operator_obj, 'kernel')) and not inspect.ismethod(getattr(operator_obj, 'kernel')):
         print(f"Error: 'kernel' method should be directly callable from the class (e.g., a static method).")
         return False

    print(f"Operator {operator_obj.__name__} is valid.")
    return True
