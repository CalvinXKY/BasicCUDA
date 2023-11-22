#include <pybind11/pybind11.h>
#include <iostream>

int add(int i = 2, int j = 3) {
    std::cout << "Add called, input numbers: i=" << i << " j=" << j << std::endl;
    return i + j;
}

PYBIND11_MODULE(functions, m) {
    m.doc() = "pybind11 example plugin";  // optional module docstring. Could be printed by python help().
    m.def("add", &add, "A function that adds two numbers");
}

