
/*
 *  zero copy using in vectorAdd case.
 *
 *  This demo code might be stale with the development of CUDA.
 *  To use the latest API operations, you could see NVIDIA guide:
 *      https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 *      https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
 *
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 */

#include "memoryOpt.h"
#include "timer.h"

__global__ void vectorAdd(const float *A, const float *B, float *C, const int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

float vectorAddViaGlobalMemory(const unsigned int numElements, const unsigned int iterNum)
{

    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float throughputInGBs = 0.0f;

    sdkCreateTimer(&timer);
    size_t memSize = numElements * sizeof(float);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate the host input vector A, B, C
    float *h_A = (float *)malloc(memSize);
    float *h_B = (float *)malloc(memSize);
    float *h_C = (float *)malloc(memSize);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the device input vector:
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_A, memSize));
    checkCudaErrors(cudaMalloc((void **)&d_B, memSize));
    checkCudaErrors(cudaMalloc((void **)&d_C, memSize));

    for (unsigned int i = 0; i < iterNum; i++) {
        sdkStartTimer(&timer);
        checkCudaErrors(cudaMemcpy(d_A, h_A, memSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, memSize, cudaMemcpyHostToDevice));
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        checkCudaErrors(cudaGetLastError());
        // Copy the device result vector in device memory to the host result vector in host memory.
        checkCudaErrors(cudaMemcpy(h_C, d_C, memSize, cudaMemcpyDeviceToHost));
        sdkStopTimer(&timer);
        elapsedTimeInMs += sdkGetTimerValue(&timer);
        sdkResetTimer(&timer);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // calculate throughput in GB/s. Note: use 1000(not 1024)unit.
    double time_s = elapsedTimeInMs / 1e3;
    throughputInGBs = (memSize * (float)iterNum) / (double)1e9;
    throughputInGBs = throughputInGBs / time_s;
    sdkDeleteTimer(&timer);

    // Free device global memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return throughputInGBs;
}

float vectorAddViaZeroCopy(const unsigned int numElements, const unsigned int iterNum)
{

    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float throughputInGBs = 0.0f;

    sdkCreateTimer(&timer);
    size_t memSize = numElements * sizeof(float);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
    // Allocate the host input vector A, B, C
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *map_A, *map_B, *map_C;
    // Policy1:
    // checkCudaErrors(cudaMallocHost((void **)&h_A, memSize));
    // checkCudaErrors(cudaMallocHost((void **)&h_B, memSize));
    // checkCudaErrors(cudaMallocHost((void **)&h_C, memSize));

    // Policy2:
    checkCudaErrors(cudaHostAlloc((void **)&h_A, memSize, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_B, memSize, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_C, memSize, cudaHostAllocMapped));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    // Get the device pointers for the pinned CPU memory mapped into the GPU memory space.
    checkCudaErrors(cudaHostGetDevicePointer(&map_A, h_A, 0));
    checkCudaErrors(cudaHostGetDevicePointer(&map_B, h_B, 0));
    checkCudaErrors(cudaHostGetDevicePointer(&map_C, h_C, 0));

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in device memory
    for (unsigned int i = 0; i < iterNum; i++) {
        sdkStartTimer(&timer);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(map_A, map_B, map_C, numElements);
        checkCudaErrors(cudaGetLastError());
        // Copy the device result vector in device memory to the host result vector in host memory.
        sdkStopTimer(&timer);
        elapsedTimeInMs += sdkGetTimerValue(&timer);
        sdkResetTimer(&timer);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // calculate throughput in GB/s. Note: use 1000(not 1024)unit.
    double time_s = elapsedTimeInMs / 1e3;
    throughputInGBs = (memSize * (float)iterNum) / (double)1e9;
    throughputInGBs = throughputInGBs / time_s;
    sdkDeleteTimer(&timer);

    // Free host memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));

    return throughputInGBs;
}

int main(int argc, char **argv)
{
    printf("[Zero Copy Opt Vector Add] - Starting...\n");
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -size=The size of numElements for testing in bytes. Default: 5000000)\n");
        printf("      -iter=n Iteration numbers of trans. Default:1 \n");
        exit(EXIT_SUCCESS);
    }
    unsigned int numElements = 5000000;
    unsigned int iterNumbers = 1;
    unsigned int gpuID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        gpuID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        numElements = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iter")) {
        iterNumbers = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
    }

    checkCudaErrors(cudaSetDevice(gpuID));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuID);
    if (!prop.canMapHostMemory)
        exit(EXIT_FAILURE);
    printf(">. Data tranfer via global memory.  VectorAdd throughput: %f GB/s\n",
           vectorAddViaGlobalMemory(numElements, iterNumbers));
    printf(">. Data tranfer via  zero copy.     VectorAdd throughput: %f GB/s\n",
           vectorAddViaZeroCopy(numElements, iterNumbers));

    exit(EXIT_SUCCESS);
}