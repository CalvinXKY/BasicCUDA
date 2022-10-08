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

## A matrix multiply in CPU
We often use x,y to describe indices of 2D Matrix, 
however its data allocated in memory is linear form 1D.
We use a data pointer to represent first location on memory. 
And transfer 2D index to 1D to get the number in matrix.
Thus, matrix multiply can be realized as follow, which is C = A x B operation on cpu. 

```c++
/*
* float *C, *A , *B: data pointer of matrix C, A, B each.
* unsigned int wA: width of A.
* unsigned int wC: width of C, which equals height of B.
* unsigned int hC: hegith of C, which equals height of A.
*/
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int wA,
                  unsigned int wC, unsigned int hC) {
  unsigned int hA = hC;
  unsigned int wB = wC;
  for (unsigned int i = 0; i < hA; ++i)
    for (unsigned int j = 0; j < wB; ++j) {
      double sum = 0;
      for (unsigned int k = 0; k < wA; ++k) {
        sum += (double)A[i * wA + k] * (double)B[k * wB + j];
      }
      C[i * wB + j] = (float)sum;
    }
}
```
The snippet shows a simple way to realize the process. It has three loops that calculate the
elements one by one.  The iterations/times of loops is hA * wB * wA. If each step costs
deltaT time and single thread is used, the total time equals: 
    
    hA * wB * wA * deltaT 
Of course there are many optimizing methods to accelerate this process on CPU.
But we turn focus on GPU, the most important idea of GPU is that has huge threads to 
deal with dense calculation scenarios. In this case, we can invoke parallel threads to
compute the "sum += (double)A[i * wA + k] * (double)B[k * wB + j];". So the total time 
cloud be theoretically reduced to:

    hA * wB * wA * deltaT / N
    
N: Depends on number of threads.  

## 1D Block Kernel

Each thread deal with data of one row  from A and one column from B to get an element of C.
e.g. Assumption the number of threads equals C size. Then thread[i][j] will finish the follow
process:

C[i][j] = Sum(A[i][k] * B[k][j]),  k = 1,2,3,4,...wA;

In fact, there threads number does not all ways equal to the element size of C,
which could less than or more than it. Thus, we need a loop to make sure when threadNum < 
sizeC, the threads group could compute all data. On the other, while threadNum > 
sizeC, need to avoid illegal memory access. The common method as follow:
    
    while (threadIdx < sizeC) { }
    
The code snippet as follow：

```cu
__global__ void MatMulKernel1D(float *C, float *A, float *B, const int wh, const int wC, const int hC)
{
    const int totalSize = wC * hC;
    int thID = threadIdx.x + blockIdx.x * blockDim.x;
    while (thID < totalSize) {
        int Cx = thID / wC;
        int Cy = thID % wC;
        float rst = 0.0;
        for (int i = 0; i < wh; i++) {
            rst += A[Cx * wh + i] * B[i * wC + Cy];
        }
        C[Cx * wC + Cy] = rst;
        thID += gridDim.x * blockDim.x;
    }
}
```

## 1D Block Kernel with Shared Memory

## 2D Block Kernel with Block Multiples Size

## 2D Block Kernel with Size Free




