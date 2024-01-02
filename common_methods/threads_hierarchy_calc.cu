/**
 *  threads hierarchy calculation example.
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

const float EPSILON = 1e-6;

bool areFloatsEqual(float a, float b) {
    return std::fabs(a - b) < EPSILON;
}

template <typename T> void check(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


__global__ void kernelAddOne3D3D(float *input, int dataNum)
{
    int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
    int i = threadInBlock + oneBlockSize*blockInGrid;
    while(i <  dataNum) {
        input[i] += 1;
        i += oneBlockSize * gridDim.x*gridDim.y*gridDim.z;
    }
}


__global__ void kernelAddOne2D2D(float *input, int dataNum)
{
    // int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    // int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    // int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
    // int i = threadInBlock + oneBlockSize*blockInGrid;
    // when:
    // threadIdx.z = 0; blockIdx.z = 0;
    // blockDim.z = 1; gridDim.z = 1;
    // then:
    // int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x;
    // int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x;
    // int oneBlockSize = blockDim.x*blockDim.y;
    int i = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*(blockIdx.x + blockIdx.y*gridDim.x);

    while(i <  dataNum) {
        input[i] += 1;
        i +=  blockDim.x*blockDim.y*gridDim.x*gridDim.y;
    }
    // thread overflow offset = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
}

__global__ void printIdx2D2D()
{
    int i = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*(blockIdx.x + blockIdx.y*gridDim.x);
    printf("Global idx %d, threadIdx.x: %d, threadIdx.y: %d threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d,  blockIdx.z: %d \n",\
    i, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

__global__ void kernelAddOne1D1D(float *input, int dataNum)
{
    // int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    // int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    // int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
    // int i = threadInBlock + oneBlockSize*blockInGrid;
    // when:
    // threadIdx.y = 0; threadIdx.z = 0; blockIdx.y= 0;  blockIdx.z = 0;
    // blockDim.y = 1; blockDim.z = 1; gridDim.y = 1; gridDim.z = 1;
    // then:
    // int threadInBlock = threadIdx.x;
    // int blockInGrid = blockIdx.x;
    // int oneBlockSize = blockDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    while(i <  dataNum) {
        input[i] += 1;
        i += blockDim.x*gridDim.x;
    }
    // thread overflow offset = blockDim.x*gridDim.x;
}

#define TOTAL_SIZE 5000
#define N 4
#define M 4
using kernel = void (*)(float *, int);

bool test(kernel func, dim3 BlocksPerGrid, dim3 threadsPerBlock) {
    unsigned int totalSize = TOTAL_SIZE;
    float* hostData = (float*) malloc(sizeof(float) * totalSize);
    float* checkData = (float*) malloc(sizeof(float) * totalSize);
    float* devicePtr;
    checkCudaErrors(cudaMalloc((void**)&devicePtr, sizeof(float) * totalSize));
    for (int i =0; i < totalSize; ++i) {
        hostData[i] = i;
        checkData[i] = i + 1;
    }
    checkCudaErrors(cudaMemcpy(devicePtr, hostData,  totalSize * sizeof(float), cudaMemcpyHostToDevice));
    func<<<BlocksPerGrid, threadsPerBlock>>>(devicePtr, totalSize);
    checkCudaErrors(cudaMemcpy(hostData, devicePtr, totalSize * sizeof(float), cudaMemcpyDeviceToHost));
    // check result:
    bool rst = true;
    for (int i =0; i < totalSize; ++i) {
        if (!areFloatsEqual(checkData[i], hostData[i])) {
            rst = false;
            printf("The result not equal in data index %d. expect:%f  result:%f\n", i, checkData[i], hostData[i]);
            break;
        }
    }
    checkCudaErrors(cudaFree (devicePtr));
    free(hostData);
    free(checkData);
    return rst;
}


int main() {
    printf("This example is for threads hierachy calculation.\n");
    // 3D3D:
    dim3 BlocksPerGrid(N, N, N);  // 对应gridDim.x、gridDim.y、gridDim.z
    dim3 threadsPerBlock(M, M, M);  // 对应blockDim.x、blockDim.y、blockDim.z
    // test(kernelAddOne3D3D, BlocksPerGrid, threadsPerBlock)

    // 2D2D:
    dim3 BlocksPerGrid2D(N, N);
    dim3 threadsPerBlock2D(M, M);
    // test(kernelAddOne2D2D, BlocksPerGrid2D, threadsPerBlock2D)

    // 1D1D:
    // test(kernelAddOne1D1D, N, M)

    // print the idx in threads, 2D2D example:
    printIdx2D2D<<<dim3(3, 3), dim3(2,2)>>>();

    bool rst = test(kernelAddOne3D3D, BlocksPerGrid, threadsPerBlock) && \
    test(kernelAddOne2D2D, BlocksPerGrid2D, threadsPerBlock2D) && \
    test(kernelAddOne1D1D, N, M);
    if(rst) {
        printf("The test OK.\n");
    } else {
        printf("The test Failed.\n");
    }
    return 0;
}
