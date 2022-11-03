/*
 *  Memory transfer between host and device example to help you understand the process.
 *
 *  This demo code might be stale with the development of CUDA.
 *  To use the latest API operations, you could see NVIDIA guide:
 *      https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 *      https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
 *
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 */

#include <cuda.h>
#include <cassert>
#include <memory>

#include "memoryOpt.h"
#include "timer.h"

#define FLUSH_SIZE (256 * 1024 * 1024)
char *flushBuf;

/*
 *  Using specific mem size to get dvice to host transfer bandwidth with pageable memory.
 */
float deviceToHostTransfer(const unsigned int memSize, const unsigned int iterNum)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    unsigned char *hInData = NULL;
    unsigned char *hOutData = NULL;
    sdkCreateTimer(&timer);

    hInData = (unsigned char *)malloc(memSize);
    hOutData = (unsigned char *)malloc(memSize);

    if (hInData == 0 || hOutData == 0) {
        fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    // initialize the memory
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        hInData[i] = (unsigned char)(i & 0xff);
    }

    // allocate device memory
    unsigned char *devInData;
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));

    // initialize the device memory
    checkCudaErrors(cudaMemcpy(devInData, hInData, memSize, cudaMemcpyHostToDevice));

    // copy data from GPU to Host
    elapsedTimeInMs = 0;
    for (unsigned int i = 0; i < iterNum; i++) {
        sdkStartTimer(&timer);
        checkCudaErrors(cudaMemcpy(hOutData, devInData, memSize, cudaMemcpyDeviceToHost));
        sdkStopTimer(&timer);
        elapsedTimeInMs += sdkGetTimerValue(&timer);
        sdkResetTimer(&timer);
        memset(flushBuf, i, FLUSH_SIZE);
    }

    // calculate bandwidth in GB/s
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)iterNum) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;
    sdkDeleteTimer(&timer);

    free(hInData);
    free(hOutData);

    checkCudaErrors(cudaFree(devInData));

    return bandwidthInGBs;
}

/*
 *  Using specific mem size to get host to device transfer bandwidth with pageable memory.
 */

float hostToDeviceTransfer(const unsigned int memSize, const unsigned int iterNum)
{
    StopWatchInterface *timer = NULL;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    sdkCreateTimer(&timer);

    // allocate host memory
    unsigned char *hOutData = NULL;
    hOutData = (unsigned char *)malloc(memSize);

    if (hOutData == 0) {
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
    unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

    if (h_cacheClear1 == 0 || h_cacheClear2 == 0) {
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    // initialize the memory
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        hOutData[i] = (unsigned char)(i & 0xff);
    }

    for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
        h_cacheClear1[i] = (unsigned char)(i & 0xff);
        h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
    }

    // allocate device memory
    unsigned char *devInData;
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));

    // copy host memory to device memory
    elapsedTimeInMs = 0;
    for (unsigned int i = 0; i < iterNum; i++) {
        sdkStartTimer(&timer);
        checkCudaErrors(cudaMemcpy(devInData, hOutData, memSize, cudaMemcpyHostToDevice));
        sdkStopTimer(&timer);
        elapsedTimeInMs += sdkGetTimerValue(&timer);
        sdkResetTimer(&timer);
        memset(flushBuf, i, FLUSH_SIZE);
    }

    // calculate bandwidth in GB/s
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)iterNum) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;
    sdkDeleteTimer(&timer);

    free(hOutData);
    free(h_cacheClear1);
    free(h_cacheClear2);
    checkCudaErrors(cudaFree(devInData));
    return bandwidthInGBs;
}

/*
 *  Using specific mem size to get host to device transfer bandwidth with pinned memory.
 */
float hostToDeviceTransferWithPinned(const unsigned int memSize, const unsigned int iterNum, bool wc)
{

    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // allocate host memory
    unsigned char *hOutData = NULL;

#if CUDART_VERSION >= 2020
    // pinned memory mode - use special function to get OS-pinned memory
    // WC memory is a good option for buffers that will be written
    // by the CPU and read by the device via mapped pinned memory or host->device transfers.
    checkCudaErrors(cudaHostAlloc((void **)&hOutData, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
#else
    // pinned memory mode - use special function to get OS-pinned memory
    checkCudaErrors(cudaMallocHost((void **)&hOutData, memSize));
#endif

    unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
    unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

    if (h_cacheClear1 == 0 || h_cacheClear2 == 0) {
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    // initialize the memory
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        hOutData[i] = (unsigned char)(i & 0xff);
    }

    for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
        h_cacheClear1[i] = (unsigned char)(i & 0xff);
        h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
    }

    // allocate device memory
    unsigned char *devInData;
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));

    // copy host memory to device memory

    checkCudaErrors(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < iterNum; i++) {
        checkCudaErrors(cudaMemcpyAsync(devInData, hOutData, memSize, cudaMemcpyHostToDevice, 0));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    // calculate bandwidth in GB/s
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)iterNum) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;
    // clean up memory
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFreeHost(hOutData));

    free(h_cacheClear1);
    free(h_cacheClear2);
    checkCudaErrors(cudaFree(devInData));

    return bandwidthInGBs;
}

/*
 *  Using specific mem size to get host to device transfer bandwidth with pinned memory.
 */
float deviceToHostTransferWithPinned(const unsigned int memSize, const unsigned int iterNum, bool wc)
{

    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // allocate host memory
    unsigned char *hOutData = NULL;

#if CUDART_VERSION >= 2020
    // pinned memory mode - use special function to get OS-pinned memory
    // WC memory is a good option for buffers that will be written
    // by the CPU and read by the device via mapped pinned memory or host->device transfers.
    checkCudaErrors(cudaHostAlloc((void **)&hOutData, memSize, (wc) ? cudaHostAllocWriteCombined : 0));
#else
    // pinned memory mode - use special function to get OS-pinned memory
    checkCudaErrors(cudaMallocHost((void **)&hOutData, memSize));
#endif

    unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
    unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

    if (h_cacheClear1 == 0 || h_cacheClear2 == 0) {
        fprintf(stderr, "Not enough memory available on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    // initialize the memory
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        hOutData[i] = (unsigned char)(i & 0xff);
    }

    for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
        h_cacheClear1[i] = (unsigned char)(i & 0xff);
        h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
    }

    // allocate device memory
    unsigned char *devInData;
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));

    // initialize the device memory
    checkCudaErrors(cudaMemcpy(devInData, hOutData, memSize, cudaMemcpyHostToDevice));

    // copy host memory to device memory

    checkCudaErrors(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < iterNum; i++) {
        checkCudaErrors(cudaMemcpyAsync(hOutData, devInData, memSize, cudaMemcpyDeviceToHost, 0));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    // calculate bandwidth in GB/s
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)iterNum) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;
    // clean up memory
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFreeHost(hOutData));

    free(h_cacheClear1);
    free(h_cacheClear2);
    checkCudaErrors(cudaFree(devInData));

    return bandwidthInGBs;
}

int main(int argc, char **argv)
{
    flushBuf = (char *)malloc(FLUSH_SIZE);
    printf("[Host and Device Memory Opt Demo:] - Starting...\n");
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -size=The size of memory for testing in bytes. Default: 20*1024*1024)\n");
        printf("      -iter=n Iteration numbers of trans. Default:100 \n");
        exit(EXIT_SUCCESS);
    }
    unsigned int memSize = 1024 * 1024 * 20;
    unsigned int iterNumber = 100;
    unsigned int gpuID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        gpuID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        memSize = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iter")) {
        iterNumber = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
    }

    checkCudaErrors(cudaSetDevice(gpuID));
    printf("Test data transfer with pageable memory\n");
    printf(">. hostToDeviceTransfer bandwith: %f GB/s\n", hostToDeviceTransfer(memSize, iterNumber));
    printf(">. deviceToHostTransfer bandwith: %f GB/s\n", deviceToHostTransfer(memSize, iterNumber));

    printf("Test data transfer with pinned memory\n");
    printf(">. hostToDeviceTransferWithPinned bandwith: %f GB/s\n",
           hostToDeviceTransferWithPinned(memSize, iterNumber, true));
    printf(">. deviceToHostTransferWithPinned bandwith: %f GB/s\n",
           deviceToHostTransferWithPinned(memSize, iterNumber, true));

    free(flushBuf);
    exit(EXIT_SUCCESS);
}