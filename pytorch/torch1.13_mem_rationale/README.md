# PyTorch Memory Cuda Allocator Test
*Objective of this Submodule*: Compiling and executing the cudaCachingAllocator derived from the PyTorch source code presents a noteworthy challenge, especially when attempting to test individual segments of the c10 cuda components. 
The primary aim of this submodule is to distill and streamline the source code to ensure it can be effortlessly executed within various testing frameworks.

## Comilple & Run

Compile:
```
./make
```
You would get an "allocator_test" exe file, then run it.

```
./allocator_test
```