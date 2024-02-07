import torch
from torch.utils.cpp_extension import load_inline

cpp_src = """
#include <torch/extension.h>

void printArray(torch::Tensor input) {
    int *ptr = (int *)input.data_ptr();
    for(int i=0; i <  input.numel(); i++) {
        printf("%d\\n", ptr[i]);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("print_array", &printArray, "");
}
"""

ext_module = load_inline(name="print_array", cpp_sources=cpp_src, verbose=True)
ext_module.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))