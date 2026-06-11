import inspect

import triton
import triton.language as tl

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

def dump_kernel_ptx(kernel):
    inputs = kernel.get_random_input()
    _, compiled_kernel = kernel.forward_triton(inputs)

    return extract_ptx(compiled_kernel)

def jit_fixed_parameters(fn=None, **triton_kwargs):
    """
    A decorator that automatically populates 'do_not_specialize' for any
    parameter NOT marked as tl.constexpr, then calls triton.jit.
    This implies that the ptx signature will keep this argument and remove 
    only tl.constexpr arguments, which are fixed at compile time and not parametric in the ptx code.
    """

    def decorator(func):
        params = inspect.signature(func).parameters
        annotations = inspect.get_annotations(func, eval_str=True)
        do_not_specialize_list = []

        for name, param in params.items():
            annotation = annotations.get(name, param.annotation)
            if annotation is not tl.constexpr:
                do_not_specialize_list.append(name)

        return triton.jit(fn=func, do_not_specialize=do_not_specialize_list, **triton_kwargs)

    if fn is None:
        return decorator

    return decorator(fn)
