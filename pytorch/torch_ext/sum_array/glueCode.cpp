#include "sumArray.h"
#include <torch/extension.h>


torch::Tensor torchSumArray(torch::Tensor input) {
    int dataSize = input.numel();
    float* devInData = (float *)input.data_ptr();
    arraySumCUDA(devInData, dataSize);
    return input[0];
}

PYBIND11_MODULE(sum_array, m) {
    m.def("sum_array", &torchSumArray, "");
}