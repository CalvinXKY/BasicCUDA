#include <cstdio>
#include "cuda_runtime.h"
#define N 8


__global__ void kernel(int mark)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("  === kernel %d run info: gridDim.x: %d, blockDim.x: %d ===\n", \
        mark, gridDim.x, blockDim.x);
    }
    __syncthreads();
    printf("    blockIdx.x: %d threadIdx.x: %d\n", blockIdx.x,  threadIdx.x);
}

__global__ void kernelCalcuDim(int dimNum)
{
    if (threadIdx.x + threadIdx.y + threadIdx.z + blockIdx.x + blockIdx.y + blockIdx.z == 0) {
        printf("============= The grid shape: gridDim.x: %d gridDim.y: %d gridDim.z: %d\n",\
        gridDim.x, gridDim.y, gridDim.z);
        printf("============= The block shape: blockDim.x: %d blockDim.y: %d blockDim.z: %d\n",\
        blockDim.x, blockDim.y, blockDim.z);
    }
    __syncthreads();
    int offset = 0;
    int x, y, z;
    switch (dimNum) {
        case 1:
            offset = threadIdx.x + blockIdx.x * blockDim.x;
            break;
        case 2:
            x = threadIdx.x + blockIdx.x * blockDim.x;
            y = threadIdx.y + blockIdx.y * blockDim.y;
            offset = x + y * blockDim.x * gridDim.x;
            // method 2:
            // offset = threadIdx.x + blockDim.x * threadIdx.y + \
            // (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);
            break;
        case 3:
            x = threadIdx.x + blockIdx.x * blockDim.x;
            y = threadIdx.y + blockIdx.y * blockDim.y;
            z = threadIdx.z + blockIdx.z * blockDim.z;
            offset = x + y * blockDim.x * gridDim.x + z * blockDim.x * blockDim.y * gridDim.x * gridDim.y;
            break;
        default:
            break;
    }

    printf("    blockIdx: x=%d y= %d z=%d threadIdx x=%d y=%d z=%d; offset= %d\n",\
    blockIdx.x, blockIdx.y, blockIdx.z,  threadIdx.x, threadIdx.y, threadIdx.z, offset);
}


int main()
{
    printf("Case0: the diff between <<<1, N>>> with <<<N, 1>>>\n");
    printf(" Kernel 0 invocation with N threads (1 blocks, N thread/block) N =%d\n" , N);
    kernel<<<1, N>>>(0);
    cudaDeviceSynchronize();
    printf(" Kernel 1 invocation with N threads (N blocks, 1 thread/block) N =%d\n" , N);
    kernel<<<N, 1>>>(1);
    cudaDeviceSynchronize();
    printf("\n\n");

    printf("Case1: 1 dimension, grid: 2  block: 2 \n");
    kernelCalcuDim<<<2, 2>>>(1);
    cudaDeviceSynchronize();
    printf("\n");

    printf("Case2: 2 dimension, grid: 2 x 1  block: 2 x 2 \n");
    dim3 gridSize2D(2, 1);
    dim3 blockSize2D(2, 2);
    kernelCalcuDim<<<gridSize2D, blockSize2D>>>(2);
    cudaDeviceSynchronize();
    printf("\n");

    printf("Case3: 3 dimension, grid: 2 x 1 x 2 block: 1 x 2 x 2 \n");
    dim3 gridSize3D(2, 1, 2);
    dim3 blockSize3D(1, 2, 2);
    kernelCalcuDim<<<gridSize3D, blockSize3D>>>(3);
    cudaDeviceSynchronize();
    return 0;
}