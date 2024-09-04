/*
 *  Ref: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 *  Compile: nvcc  -lnccl -ccbin g++ -std=c++11 -O3 -g multi_devices_per_thread.cu -o multi_devices_per_thread
 */

#include "comm.h"

int main(int argc, char *argv[])
{
    ncclComm_t comms[DEFAULT_DEVICES_NUM];
    int nDev = DEFAULT_DEVICES_NUM;
    int size = 1024 * 1024;

    // allocating and initializing device buffers
    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, gpu_ids));

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        NCCLCHECK(
            ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);

    printf("Success \n");
    return 0;
}