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

Add GPU architecture: change SM in Makefile: L255

```shell
SMS ?= 35 37 50 52 60 61 70 75 80 86 90
```
eg. SMS=80 means supports A100/A800 

SMS=90, H100/H800


Note: Your version CUDA nvcc might get "unsupported gpu architecture 'compute_35'" error. Delete SMS 35.
