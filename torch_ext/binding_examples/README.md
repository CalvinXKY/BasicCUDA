

## Step One (Basic examples)

We connect c++ code with python calling through a lib: pybind11. Before you start, make sure it's installed correctly

Simple way:
```shell
pip install pybind11
```
### Case1: How to call function.
Related files in "step_one file":
* “functions.cc". Your c++ function implementation.  
* ”function_call.py“. A demo shows call the c++ functions.  

pybind11 provides a way connect c++ with python:
```c++
PYBIND11_MODULE(functions, m) {
    m.doc() = "pybind11 example plugin";  // optional module docstring. Could be printed by python help().
    m.def("add", &add, "A function that adds two numbers");
}
```
parameter explain:
* PYBIND11_MODULE() macro: Create functions and classes for python calling.
* functions: The module created to import in python env.
* m: A variable of type py::module_ which is the main interface for creating bindings.
* .doc : Define module docstring.
* .def : Define a python func for calling.

Running the snippet:
1 Use c++ compiling a python lib.
functions.
```shell
# Compiler: g++/gcc.
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) functions.cc -o functions.so
```

2 To run ”function_call.py“, you'll see info below:

```
Add called, input numbers: i=3 j=4
7
```

**Note:**
pybind11 python pkg provides binding libs. You can see them in its lib.
```shell
echo $(python3 -m pybind11 --includes)
```
Printing info on console likes:
```
-I/home/kaiyuan/anaconda3/envs/py3.9/include/python3.9 -I/home/kaiyuan/anaconda3/envs/py3.9/lib/python3.9/site-packages/pybind11/include
```

### Case2: How to call objects.

This example shows c++ class basic calling and how to deal with overload and inheritance.

Related files in "step_one file":
* “classes.cc". Your c++ classes implementation.
* ”objects_call.py“. A demo shows call the c++ obj.  

1. Compile to get .so:
```shell
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) classes.cc -o classes.so
```

2. To run ”objects_call.py“, you'll see info below:

```
The profile of this shape:
Desription:Basic example of c++ class
Num:1
Area:10.000000
The profile of this shape:
Desription:Basic example of c++ class
Num:2
Area:20.000000
100.0
```