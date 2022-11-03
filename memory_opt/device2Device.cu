/*
 *  Memory transfer between device and device example to help you understand the process.
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

/*
 *  transfer data in device itself.
 */
float deviceToItself(const unsigned int memSize, const unsigned int iterNum)
{

    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    unsigned char *devInData;
    unsigned char *devOutData;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // allocate host memory
    unsigned char *hInData = (unsigned char *)malloc(memSize);

    if (hInData == 0) {
        fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        hInData[i] = (unsigned char)(i & 0xff);
    }

   
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));
    checkCudaErrors(cudaMalloc((void **)&devOutData, memSize));
    checkCudaErrors(cudaMemcpy(devInData, hInData, memSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start, 0));

    for (unsigned int i = 0; i < iterNum; i++) {
        checkCudaErrors(cudaMemcpy(devOutData, devInData, memSize, cudaMemcpyDeviceToDevice));
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    // In and Out, mutilpy 2.0 factor. Note: use 1000(not 1024)unit.
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (2.0f * memSize * (float)iterNum) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    free(hInData);
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFree(devInData));
    checkCudaErrors(cudaFree(devOutData));
    return bandwidthInGBs;
}

/*
 *  transfer data from one devcie to another without peer-to-peer opt.
 */
float deviceToDeviceWithoutP2P(const unsigned int memSize, const unsigned int iterNum, const unsigned int GPUA,
                               const unsigned int GPUB)
{

    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    unsigned char *devInData;
    unsigned char *devOutData;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // allocate host memory
    unsigned char *hInData = (unsigned char *)malloc(memSize);

    if (hInData == 0) {
        fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        hInData[i] = (unsigned char)(i & 0xff);
    }

    cudaSetDevice(GPUA);
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));
    cudaSetDevice(GPUB);
    checkCudaErrors(cudaMalloc((void **)&devOutData, memSize));
    cudaSetDevice(GPUA);
    checkCudaErrors(cudaMemcpy(devInData, hInData, memSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < iterNum; i++) {
        checkCudaErrors(cudaMemcpy(devOutData, devInData, memSize, cudaMemcpyDeviceToDevice));
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    // In and Out. Note: use 1000(not 1024)unit.
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)iterNum) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    free(hInData);
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFree(devInData));
    checkCudaErrors(cudaFree(devOutData));
    return bandwidthInGBs;
}

/*
 *  transfer data from one devcie to another with peer-to-peer opt.
 */
float deviceToDeviceWithP2P(const unsigned int memSize, const unsigned int iterNum, const unsigned int GPUA,
                            const unsigned int GPUB)
{

    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    unsigned char *devInData;
    unsigned char *devOutData;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // allocate host memory
    unsigned char *hInData = (unsigned char *)malloc(memSize);

    if (hInData == 0) {
        fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        hInData[i] = (unsigned char)(i & 0xff);
    }
    checkCudaErrors(cudaSetDevice(GPUA));
 
    // enable GPUA access GPUB
    checkCudaErrors(cudaDeviceEnablePeerAccess(GPUB, 0));
    checkCudaErrors(cudaMalloc((void **)&devInData, memSize));
    checkCudaErrors(cudaSetDevice(GPUB));
    // enable GPUB access GPUA
    checkCudaErrors(cudaDeviceEnablePeerAccess(GPUA, 0));
    checkCudaErrors(cudaMalloc((void **)&devOutData, memSize));
    checkCudaErrors(cudaSetDevice(GPUA));
    checkCudaErrors(cudaMemcpy(devInData, hInData, memSize, cudaMemcpyHostToDevice));


    checkCudaErrors(cudaEventRecord(start, 0));
    for (unsigned int i = 0; i < iterNum; i++) {
        checkCudaErrors(cudaMemcpy(devOutData, devInData, memSize, cudaMemcpyDeviceToDevice));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    // In and Out. Note: use 1000(not 1024)unit.
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)iterNum) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    free(hInData);
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFree(devInData));
    checkCudaErrors(cudaFree(devOutData));
    return bandwidthInGBs;
}

int main(int argc, char **argv)
{
    printf("[Device to Device Memory Opt Demo:] - Starting...\n");
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -deviceA=n (n >= 0 for deviceID A. Default:0)\n");
        printf("      -deviceB=n (n >= 0 for deviceID B. Default:1)\n");
        printf("      -size=The size of memory for testing in bytes. Default: 20*1024*1024)\n");
        printf("      -iter=n Iteration numbers of trans. Default:100 \n");
        exit(EXIT_SUCCESS);
    }
    unsigned int memSize = 1024 * 1024 * 20;
    unsigned int iterNumbers = 100;
    unsigned int GPUA = 0;
    unsigned int GPUB = 1;

    if (checkCmdLineFlag(argc, (const char **)argv, "deviceA")) {
        GPUA = getCmdLineArgumentInt(argc, (const char **)argv, "deviceA");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "deviceB")) {
        GPUB = getCmdLineArgumentInt(argc, (const char **)argv, "deviceB");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        memSize = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iter")) {
        iterNumbers = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
    }
    checkCudaErrors(cudaSetDevice(GPUA));
    printf(">. Device to itself transfer.             Bandwith: %f GB/s\n", deviceToItself(memSize, iterNumbers));
    printf(">. Device to device transfer without p2p. Bandwith: %f GB/s\n",
           deviceToDeviceWithoutP2P(memSize, iterNumbers, GPUA, GPUB));
    printf(">. Device to device transfer with p2p     Bandwith: %f GB/s\n",
           deviceToDeviceWithP2P(memSize, iterNumbers, GPUA, GPUB));

    exit(EXIT_SUCCESS);
}