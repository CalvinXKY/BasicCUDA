# PyTorch Extension Custom C++/CUDA 



## easyJIT

Requirements:
1. Pytorch (> 1.8 is better)
2. Ninja
3. CUDA


Run:
```
cd easyJIT
python run.py
```

Common issues:
1. RuntimeError: Ninja is required to load C++ extensions

Solution: pip install Ninja

This example implement a custom c++ function to print tensor array.
The lines of the code is less than 20. It's the first step make you
know the process. The key elements as follows:

1. pybind11: binding custom code to python.
2. #include <torch/extension.h> : includes pytorch defined func/param/kernel. e.g. torch::tensor
3. from torch.utils.cpp_extension import load: call Ninja JIT to compile the code and import it to python.

## easySetup

Run:
```
cd easySetup
python setup install
python run.py
```
Use setup method does not need to compile code every time. It installs the extension as a
python module. 

## sumArray

Run:
```
cd sumArray
python run.py
```

This example shows how to use CUDA kernel to accomplish custom sum of a tensor array.
You might find custom one runs faster than torch.sum().

Result likes:

```
...
Loading extension module sum_array...
tensor(24969.7930, device='cuda:0')
tensor(24969.7930, device='cuda:0')
The torch original sum func test:
    Elapsed time: 0.07710027694702148
The custom define sum func test:
    Elapsed time: 0.06388998031616211
```

