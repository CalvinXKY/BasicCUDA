
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

/**
 * Function source code
 *
 * **/

int addTwoNum(int a, int b){
    return a + b;
}

/**
 * overload functions
 * **/

void printInfo(int digis) {
   std::cout << "Your input is integer:" << std::to_string(digis) << std::endl;
}

void printInfo(float digis) {
    std::cout << "Your input is string:" << std::to_string(digis) << std::endl;
}

/**
* inplace case:
**/

void inplaceAdd(int& src, int increment) {
    src += increment;
}

struct Data{
    int num=0;
};

void inplaceAddV2(Data& data, int increment) {
    data.num += increment;
}

void setDataPtr100(Data* data) {
    data->num = 100;
}

/**
* global variable:
**/
int worldCount = 9;

/**
 * template function
 * **/
template <typename T>
T multiply(const T& a, const T& b) {
    return a * b;
}

/**
 * Allow/Prohibiting None arguments
 * **/
void showDataNum(Data* data) {
    if (data) {
        std::cout << "The data.num:" << data->num << std::endl;
        return;
    }
    std::cout << "No data input" << std::endl;
}

/**
 * recall function
 * **/

typedef int (*FUN)(int);

int addOne(int a){
    a += 1;
    return a;
}

void recallFunc(FUN f) {
    int a = 10;
    a = f(a);
}


PYBIND11_MODULE(functions, m) {
    m.def("add_two_num", &addTwoNum, "Input int a and int b,return a + b");
    m.def("add_two_num_with_default",  &addTwoNum, "default a=1, b=2", py::arg("a")=1, py::arg("b")=2);
    m.def("printInfo", static_cast<void (*)(float)>(&printInfo), "Overload examples", py::arg("digis"));
    m.def("inplace_add", &inplaceAdd, "Expect: input(&a, b), a += b, but it does not work.");
    m.def("inplace_add_use_struct", &inplaceAddV2, "data.num += b");
    py::class_<Data>(m, "Data")
      .def(py::init<>())
      .def_readwrite("num", &Data::num);
    m.def("set_data_ptr_100", &setDataPtr100, "data->num= 100");
    m.attr("worldCount")=worldCount;
    m.def("multiply", &multiply<float>);
    m.def("multiply", &multiply<int>);
    m.def("multiply_float", &multiply<double>, py::arg("a").noconvert(), py::arg("b").noconvert());
    m.def("show_data_num", &showDataNum, py::arg("data").none(false));
    m.def("show_data_num_allow_none", &showDataNum, py::arg("data").none(true));
}

