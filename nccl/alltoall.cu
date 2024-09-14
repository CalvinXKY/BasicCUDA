/*
 *  Compile: nvcc  -lnccl -ccbin g++ -std=c++11 -O3 -g alltoall.cu -o alltoall
 *  Test: ./alltoall
 *  Profiling: nvprof --csv -o profile_output.csv ./alltoall
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 */

#include "comm.h"

ncclUniqueId id;
pthread_mutex_t mutex;

void dataPrint(float *hostData, int size, int gpu_id, int my_nranks, const char *status)
{
    pthread_mutex_lock(&mutex);
    printf("GPU:%d %s data: ", gpu_id, status);
    for (int i = 0; i < size; ++i) {
        printf("%.0f ", hostData[i]);
    }
    printf("\n");
    pthread_mutex_unlock(&mutex);
}

ncclResult_t AlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                      cudaStream_t stream)
{
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * wordSize(type);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return ncclInternalError;
#else
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; r++) {
        NCCLCHECK(ncclSend(((char *)sendbuff) + r * rankOffset, count, type, r, comm, stream));
        NCCLCHECK(ncclRecv(((char *)recvbuff) + r * rankOffset, count, type, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
#endif
}

void *threadAlltoAll(void *arg)
{
    int count = 1;
    int size = my_nranks * count;
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
        // hostData[i] = float(gpu_id) * my_nranks + i;
        hostData[i] = float(gpu_id);
    }
    dataPrint(hostData, size, gpu_id, my_nranks, "input");

    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    cudaMemcpy(sendbuff, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDACHECK(cudaStreamCreate(&s));

    NCCLCHECK(AlltoAll((const void *)sendbuff, (void *)recvbuff, count, ncclFloat, comm, s));
    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));
    cudaMemcpy(hostData, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost);
    dataPrint(hostData, size, gpu_id, my_nranks, "output");
    ncclCommDestroy(comm);

    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    free(hostData);

    return NULL;
}

void *threadAlltoAllIter(void *arg)
{
    int count = 2 * 1024 * 1024;
    int size = my_nranks * count;
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
        // hostData[i] = float(gpu_id) * my_nranks + i;
        hostData[i] = float(gpu_id);
    }
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    cudaMemcpy(sendbuff, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDACHECK(cudaStreamCreate(&s));

    for (int i = 0; i < 4; ++i) {
        NCCLCHECK(AlltoAll((const void *)sendbuff, (void *)recvbuff, count, ncclFloat, comm, s));
        // Sync stream to avoid data chaos.
        CUDACHECK(cudaStreamSynchronize(s));
    }

    cudaMemcpy(hostData, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost);
    ncclCommDestroy(comm);

    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    free(hostData);
    return NULL;
}

void runAlltoAll(ops threadFunc)
{
    pthread_t threads[8];
    printf("====== AlltoAll case begin =====\n");
    NCCLCHECK(ncclGetUniqueId(&id));
    for (int i = 0; i < my_nranks; ++i) {
        int *id_pointer = &gpu_ids[i];
        pthread_create(&threads[i], NULL, threadFunc, id_pointer);
    }

    for (int i = 0; i < my_nranks; ++i) {
        pthread_join(threads[i], NULL);
    }
    printf("====== AlltoAll case end =====\n\n");
}

ncclResult_t AlltoAllSplit(const void *sendbuff, void *recvbuff, const size_t *sendSplitList,
                           const size_t *recvSplitList, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream)
{
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t sendOffset = 0;
    size_t recvOffset = 0;
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; r++) {
        NCCLCHECK(ncclSend(((char *)sendbuff) + sendOffset, sendSplitList[r], type, r, comm, stream));
        NCCLCHECK(ncclRecv(((char *)recvbuff) + recvOffset, recvSplitList[r], type, r, comm, stream));
        sendOffset += wordSize(type) * sendSplitList[r];
        recvOffset += wordSize(type) * recvSplitList[r];
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
}

const int countTotal = 15;
const size_t sendArray[4][4] = {{1, 2, 3, 4}, {4, 2, 3, 1}, {3, 2, 1, 4}, {2, 3, 4, 1}};

const size_t recvArray[4][4] = {{1, 4, 3, 2}, {2, 2, 2, 3}, {3, 3, 1, 4}, {4, 1, 4, 1}};

/*
input data:
GPU:0 : 0 0 0 0 0 0 0 0 0 0
GPU:1 : 1 1 1 1 1 1 1 1 1 1
GPU:2 : 2 2 2 2 2 2 2 2 2 2
GPU:3 : 3 3 3 3 3 3 3 3 3 3

split array:
sendArray set to:
{{1, 2, 3, 4},
{4, 2, 3, 1},
{3, 2, 1, 4},
{2, 3, 4, 1}};
recvArray is equals to transpose(sendArray)：
{{1, 4, 3, 2},
{2, 2, 2, 3},
{3, 3, 1, 4},
{4, 1, 4, 1}};

output data：
GPU:0 : 0 1 1 1 1 2 2 2 3 3
GPU:1 : 0 0 1 1 2 2 3 3 3
GPU:2 : 0 0 0 1 1 1 2 3 3 3 3
GPU:3 : 0 0 0 0 1 2 2 2 2 3


 */
void *threadAlltoAllSplit(void *arg)
{
    int size = countTotal;
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
    int sendDataNum = 0;
    int recvDataNum = 0;
    for (int i = 0; i < 4; ++i) {
        sendDataNum += sendArray[gpu_id][i];
        recvDataNum += recvArray[gpu_id][i];
    }

    dataPrint(hostData, sendDataNum, gpu_id, my_nranks, "input");

    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    cudaMemcpy(sendbuff, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDACHECK(cudaStreamCreate(&s));

    NCCLCHECK(AlltoAllSplit((const void *)sendbuff, (void *)recvbuff, sendArray[gpu_id], recvArray[gpu_id], ncclFloat,
                            comm, s));
    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));
    cudaMemcpy(hostData, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost);
    dataPrint(hostData, recvDataNum, gpu_id, my_nranks, "output");
    ncclCommDestroy(comm);
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    free(hostData);
    return NULL;
}

void runAlltoAllSplit()
{
    pthread_t threads[4];
    NCCLCHECK(ncclGetUniqueId(&id));
    if (my_nranks < 4) {
        printf("AlltoAllSplit demo requires nranks>=4, but got %d.\n", my_nranks);
        exit(-1);
    }
    // only support 4 ranks demo.
    my_nranks = 4;
    printf("====== AlltoAllSplit case begin =====\n");
    for (int i = 0; i < my_nranks; ++i) {
        int *id_pointer = &gpu_ids[i];
        pthread_create(&threads[i], NULL, threadAlltoAllSplit, id_pointer);
    }
    for (int i = 0; i < my_nranks; ++i) {
        pthread_join(threads[i], NULL);
    }
    printf("====== AlltoAllSplit case end =====\n\n");
}

int main(int argc, char *argv[])
{
    env_init(argc, argv);
    runAlltoAll(threadAlltoAll);
    runAlltoAll(threadAlltoAllIter);
    runAlltoAllSplit();
    printf("Finished successfully.\n");
    return 0;
}
