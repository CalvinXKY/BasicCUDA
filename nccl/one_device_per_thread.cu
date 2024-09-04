/*
 *  Ref: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 *  Compile: nvcc  -lnccl -ccbin g++ -std=c++11 -O3 -g one_devices_per_thread.cu.cu -o one_devices_per_thread
 */

#include "comm.h"

ncclUniqueId id;

void *thread_function(void *arg)
{
    int size = 32 * 1024 * 1024;
    int gpu_id = *(int *)arg;
    cudaSetDevice(gpu_id);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, my_nranks, id, gpu_id));

    float *sendbuff;
    float *recvbuff;
    float *hostData;
    cudaStream_t s;

    hostData = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
        hostData[i] = float(gpu_id);
    }

    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    cudaMemcpy(sendbuff, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDACHECK(cudaStreamCreate(&s));

    NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size, ncclFloat, ncclSum, comm, s));
    cudaMemcpy(hostData, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU:%d data: %f.\n", gpu_id, hostData[1]);

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);

    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    free(hostData);

    return NULL;
}

int main(int argc, char *argv[])
{
    env_init(argc, argv);
    pthread_t threads[8];
    NCCLCHECK(ncclGetUniqueId(&id));
    for (int i = 0; i < my_nranks; ++i) {
        int *id_pointer = &gpu_ids[i];
        pthread_create(&threads[i], NULL, thread_function, id_pointer);
    }

    for (int i = 0; i < my_nranks; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("Finished successfully.\n");
    return 0;
}