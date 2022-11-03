import torch
from torch.utils.cpp_extension import load_inline

cpp_src = """
#include <torch/script.h>

void printArray(torch::Tensor input) {
    int *ptr = (int *)input.data_ptr();
    for(int i = 0; i <  input.numel(); i++) {
        printf("%d\\n", ptr[i]);
    }
}

void printReverseArray(torch::Tensor input) {
    int *ptr = (int *)input.data_ptr();
    for(int i = input.numel()-1; i >= 0; --i) {
        printf("%d\\n", ptr[i]);
    }
}

static auto registry = torch::RegisterOperators("new_ops::print_array", &printArray)
                       .op("new_ops::print_reverse_array", &printReverseArray);
"""

load_inline(name="print_array", cpp_sources=cpp_src, is_python_module=False, verbose=True)
torch.ops.new_ops.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
print("Reverse:")
torch.ops.new_ops.print_reverse_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
