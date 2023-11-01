
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

class Shape {
    int num;
    float area;

   public:
    std::string profile;
    Shape(int num, float area, const std::string& profile) : num(num), area(area), profile(profile) {}
    void setProperty(int num_) { num = num_; }
    void setProperty(float area_) { area = area_; }  // function overload
    float getArea() const { return area; }
    std::string getProfile() const { return "Desription:" + profile + "\nNum:" + std::to_string(num) + "\nArea:" + std::to_string(area); }
};

class Rectangle : public Shape {
    int length;
    int width;
   public:
    Rectangle(int num, int length, int width, const std::string& profile)
        : Shape(num, length * width, profile), length(length), width(width) {}
    void resetSize(int length, int width) {
        this->length = length;
        this->width = width;
    }
};

PYBIND11_MODULE(classes, m) {
    py::class_<Shape>(m, "Shape")
        .def(py::init<int, float, const std::string&>())
        .def("setProperty", static_cast<void (Shape::*)(int)>(&Shape::setProperty), "Set the shape property of num.")
        .def("setProperty", static_cast<void (Shape::*)(float)>(&Shape::setProperty), "Set the shape property of area.")
        .def("__repr__", [](Shape& shape) { return "The profile of this shape:\n" + shape.getProfile(); });
    py::class_<Rectangle, Shape>(m, "Rectangle")
        .def(py::init<int, int, int, const std::string&>())
        .def("getArea", static_cast<float (Rectangle::*)() const>(&Rectangle::getArea), "Get the area.")
        .def("resetSize", static_cast<void (Rectangle::*)(int, int)>(&Rectangle::resetSize), "Reset size of rectangle.")
        .def_readwrite("profile", &Rectangle::profile);
}
