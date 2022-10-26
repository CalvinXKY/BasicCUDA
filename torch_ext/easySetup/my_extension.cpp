#include <torch/extension.h>

void printArray(torch::Tensor input) {
    int *ptr = (int *)input.data_ptr();
    for(int i=0; i <  input.numel(); i++) {
        printf("%d\n", ptr[i]);
    }
}

PYBIND11_MODULE(my_extension, m) {
    m.def("print_array", &printArray, "");
}
