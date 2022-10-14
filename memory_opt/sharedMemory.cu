/**
 *  Array sum calculation with or without shared memory in CUDA kernel.
 *
 *  This demo code might be stale with the development of CUDA.
 *  To use the latest API operations, you could see NVIDIA guide:
 *     https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 *     https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
 *
 *   Author: kevin.xie
 *   Email: kaiyuanxie@yeah.net
 * */

#include <memory>

#include "memoryOpt.h"
#include "timer.h"

#define THREAD_PER_BLOCK 256

double sumArrayInBlockCPU(float *arrData, const unsigned int dataSize)
{
    /*  This function might help you understand the process of CUDA array sum. */
    float *blockData = (float *)calloc(dataSize / THREAD_PER_BLOCK, sizeof(float));
    int blockSize = dataSize / THREAD_PER_BLOCK; // get integer part
    int idxMax = blockSize * THREAD_PER_BLOCK;

    // Split the array into blocks and sum the blocks one by one.
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {
            int idx = i * THREAD_PER_BLOCK + j;
            while (idx < dataSize) {
                blockData[i] += arrData[idx];
                idx += idxMax;
            }
        }
    }

    double rst = 0.0;
    // sum the all blocks result;
    for (int i = 0; i < blockSize; ++i) {
        rst += blockData[i];
    }
    return rst;
}

__device__ int countSHM = 0;
__global__ void arraySumWithSHMKernel(float *arrData, const int dataSize)
{
    __shared__ float shm[THREAD_PER_BLOCK];
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (thIdx == 0) {
        countSHM = 0;
        __threadfence();
    }
    float val = 0.0;
    while (thIdx < dataSize) {
        val += arrData[thIdx];
        thIdx += blockDim.x * gridDim.x;
    }
    shm[threadIdx.x] = val;
    __syncthreads();

    for (int i = THREAD_PER_BLOCK / 2; i >= 1; i /= 2) {
        if (threadIdx.x < i)
            shm[threadIdx.x] += shm[threadIdx.x + i];
        __syncthreads();
    }

    __syncthreads();
    bool isLast = false;
    thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x == 0) {
        arrData[blockIdx.x] = shm[0];
        __threadfence();
        int value = atomicAdd(&countSHM, 1);
        isLast = (value == gridDim.x - 1);
    }
    isLast = __syncthreads_or(isLast);
    if (isLast) {
        shm[threadIdx.x] = threadIdx.x < gridDim.x ? arrData[threadIdx.x] : 0;
        __syncthreads();
        for (int i = THREAD_PER_BLOCK / 2; i >= 1; i /= 2) {
            if (threadIdx.x < i)
                shm[threadIdx.x] += shm[threadIdx.x + i];
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0)
            arrData[0] = shm[0];
    }
    __syncthreads();
}

__global__ void arraySumKernel(float *arrData, float *oData, const int dataSize)
{
    // The function needed to run twice if dataSize > threads per block.

    int thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0;
    while (thIdx < dataSize) {
        val += arrData[thIdx];
        thIdx += blockDim.x * gridDim.x;
    }
    thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    arrData[thIdx] = val;
    __syncthreads();

    // Reduce process:
    for (int i = THREAD_PER_BLOCK / 2; i >= 1; i /= 2) {
        if (threadIdx.x < i)
            arrData[thIdx] += arrData[thIdx + i];
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        oData[blockIdx.x] = arrData[thIdx];
    }
}

float sumArrayGPU(const unsigned int dataSize, unsigned int iterNumber, bool useSHM)
{
    int memSize = sizeof(float) * dataSize;
    float *hInData = (float *)malloc(memSize);
    if (hInData == 0) {
        fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    // Get the correct result for verifying.
    double sum = sumArrayInBlockCPU(hInData, dataSize);

    float *devInData, *devOutData;
    float devRst;
    float elapsedTimeInMs = 0.0f;
    if (!useSHM) {
        checkCudaErrors(cudaMalloc((void **)&devOutData, max(dataSize / THREAD_PER_BLOCK, THREAD_PER_BLOCK)));
    }
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));
    checkCudaErrors(cudaMemcpy(devInData, hInData, memSize, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;

    for (int i = 0; i < iterNumber; i++) {
        float onceTime = 0.0;
        checkCudaErrors(cudaMemcpy(devInData, hInData, memSize, cudaMemcpyHostToDevice));
        if (useSHM) {
            TIME_ELAPSE((arraySumWithSHMKernel<<<dataSize / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(devInData, dataSize)),
                        onceTime, start, stop);
        } else {
            // Run twice to get the result.
            TIME_ELAPSE(
                (arraySumKernel<<<dataSize / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(devInData, devOutData, dataSize)),
                onceTime, start, stop);
            elapsedTimeInMs += onceTime;
            TIME_ELAPSE((arraySumKernel<<<1, THREAD_PER_BLOCK>>>(devOutData, devOutData, dataSize / THREAD_PER_BLOCK)),
                        onceTime, start, stop);
        }
        checkCudaErrors(cudaDeviceSynchronize());
        elapsedTimeInMs += onceTime;
    }

    if (useSHM) {
        checkCudaErrors(cudaMemcpy(&devRst, devInData, sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        checkCudaErrors(cudaMemcpy(&devRst, devOutData, sizeof(float), cudaMemcpyDeviceToHost));
    }

    if (fabs(devRst - sum) > 1.e-6) {
        printf("Result error! GPU: %f CPU: %f\n", devRst, sum);
        exit(EXIT_FAILURE);
    }
    free(hInData);
    checkCudaErrors(cudaFree(devInData));
    if (!useSHM) {
        checkCudaErrors(cudaFree(devOutData));
    }

    return elapsedTimeInMs / iterNumber;
}

int main(int argc, char **argv)
{
    printf("[Shared Memory Application: Array Sum.] - Starting...\n");
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -size=The size of numElements for testing in bytes. Default: 5000)\n");
        printf("      -iter=n Iteration numbers of trans. Default:100 \n");
        printf("Note: The size has a limitation. Consider float type range.)\n");
        exit(EXIT_SUCCESS);
    }
    unsigned int numElements = 5000;
    unsigned int gpuID = 0;
    unsigned int iterNumber = 100;

    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        gpuID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        numElements = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }
    if (numElements < 256 || numElements > 10000) {
        printf("The size of numElements is not allowed! Support range:256~10000.\n");
        printf("You could modify the source code to extend the range.\n");
        exit(EXIT_FAILURE);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "iter")) {
        iterNumber = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
    }

    checkCudaErrors(cudaSetDevice(gpuID));
    printf("Sum array with shared memory.       Elapsed time: %f ms \n", sumArrayGPU(numElements, iterNumber, true));
    printf("Sum array without shared memory.    Elapsed time: %f ms \n", sumArrayGPU(numElements, iterNumber, false));

    exit(EXIT_SUCCESS);
}