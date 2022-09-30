# matMul - Matrix Multiplication

## Build and Run
```
$ cd <dir>
$ make
```

*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`. A100 set SMS=80
```
$ make SMS="50 60"
```

Run the matMul:

Get help information:
```
$ ./matMul help
Usage -device=n (n >= 0 for deviceID)
      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)
      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)
      -iter=n Iteration numbers of algorithm. Default:500
      -algo=[0|1|2|3|4|5] 0: Test all, 1: MatMul_1D_KERENL, 2:MatMul_1D_KERNEL_WITH_SHARED_MEMORY, 3: MatMul_2D_KERENEL_BLOCK_MULTIPLES_SIZE, 4: MatMul_2D_KERNEL_ANY_SIZE 
      5: MatMul_CUBLAS_SGEMM_KERNEL
Note: Outer matrix dimensions of A & B matrices must be equal.

```

Given the width and height of matrix with a specific algorithm e.g. 
```
$ ./matMul wA=1000 hA=312 wB=11 hB=1000 algo=4

[Matrix Multiply Test] - Starting...

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
MatrixA(1000,312), MatrixB(11,1000)
Computing result using MatMul2DTest Kernel.
Spport any size, e.g. wA=1000 hA=312 wB=11 hB=1000.
Warmup  operation done
Performance= 111.46 GFlop/s, Time= 0.062 msec, Size= 6864000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
```


## Description
This sample implements matrix multiplication.  It has different versions of realization.

Matrix 