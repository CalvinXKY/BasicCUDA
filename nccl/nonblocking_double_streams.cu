/*
 *  Implement an non-blocking example of overlapping communication
 *  Compile: nvcc  -lnccl -ccbin g++ -std=c++11 -O3 -g nonblocking_double_streams.cu -o nonblocking_double_streams
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 */

#include <sys/time.h>

#include "comm.h"

void *allReduceOps(void *args);

struct ThreadArgs {
    int gpu_id;
    int global_size;
    ncclUniqueId *id;
    int uuid = -1;
    ThreadArgs(int gpu_id, int global_size, ncclUniqueId *id)
        : gpu_id(gpu_id)
        , id(id)
        , global_size(global_size) {};
    ThreadArgs(int gpu_id, int global_size, ncclUniqueId *id, int uuid)
        : gpu_id(gpu_id)
        , id(id)
        , global_size(global_size)
        , uuid(uuid) {};
};

bool cmpID(ncclUniqueId *id1, ncclUniqueId *id2)
{
    if (memcmp(id1->internal, id2->internal, sizeof(id2->internal)) == 0) {
        printf("id1:%p is same with id2:%p\n", id1, id2);
        return false;
    } else {
        for (int i = 0; i < 128; i++) {
            char id1_ch = (id1->internal)[i];
            char id2_ch = (id2->internal)[i];
            if (id1_ch != id2_ch)
                printf("Id diff internal idx_%d: id1:%c id2:%c\n", i, id1_ch, id2_ch);
        }
        return true;
    }
}

void printTimestamp(int gpu_id, int uuid, const char *s)
{
    struct timeval now;
    struct tm timeinfo;
    if (gettimeofday(&now, NULL) == -1) {
        perror("gettimeofday");
    }
    localtime_r(&(now.tv_sec), &timeinfo);
    char time_string[80];
    strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", &timeinfo);

    char time_string_with_ms[100];
    snprintf(time_string_with_ms, sizeof(time_string_with_ms), "%s.%03ld", time_string, (long)now.tv_usec / 1000);
    printf("Group: %d GPU idx: %d. The %s time: %s\n", uuid, gpu_id, s, time_string_with_ms);
}

class CommExec {
    int gpu_nums;
    cudaStream_t s;
    pthread_t threads[8];
    bool end_flag = true;

public:
    ncclUniqueId *id_ref;
    CommExec(int gpu_nums)
        : gpu_nums(gpu_nums)
    {
    }

    void launch(ncclUniqueId &id, ops func)
    {
        NCCLCHECK(ncclGetUniqueId(&id));
        id_ref = &id;
        for (int i = 0; i < gpu_nums; i++) {
            ThreadArgs *args = new ThreadArgs(i, gpu_nums, &id);
            pthread_create(&threads[i], NULL, func, (void *)args);
        }
        end_flag = false;
    }

    void launch(ncclUniqueId &id, int uuid, ops func)
    {
        NCCLCHECK(ncclGetUniqueId(&id));
        id_ref = &id;
        for (int i = 0; i < gpu_nums; i++) {
            ThreadArgs *args = new ThreadArgs(i, gpu_nums, &id, uuid);
            pthread_create(&threads[i], NULL, func, (void *)args);
        }
        end_flag = false;
    }

    ncclUniqueId *get()
    {
        return id_ref;
    }

    void wait()
    {

        for (int i = 0; i < gpu_nums; ++i) {
            pthread_join(threads[i], NULL);
        }
        end_flag = true;
    }

    ~CommExec()
    {
        if (!end_flag)
            wait();
    }
};

void *allReduceOps(void *args)
{
    size_t size = 2e9;
    ThreadArgs *threadArgs = (struct ThreadArgs *)args;
    int gpu_id = threadArgs->gpu_id;
    cudaSetDevice(gpu_id);
    ncclComm_t comm;
    ncclResult_t state;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;

    ncclCommInitRankConfig(&comm, threadArgs->global_size, *(threadArgs->id), gpu_id, &config);
    do {
        NCCLCHECK(ncclCommGetAsyncError(comm, &state));
    } while (state == ncclInProgress);

    float *sendbuff;
    float *recvbuff;
    float *hostData;
    cudaStream_t s;

    hostData = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < 20; ++i) {
        hostData[i] = float(gpu_id);
    }

    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    cudaMemcpy(sendbuff, hostData, size * sizeof(float), cudaMemcpyHostToDevice);

    CUDACHECK(cudaStreamCreate(&s));
    CUDACHECK(cudaDeviceSynchronize());
    if (if_debug)
        printTimestamp(gpu_id, threadArgs->uuid, "start");
    NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size, ncclFloat, ncclSum, comm, s));
    // In non-blocking mode, the elapsed time has no reference.
    if (if_debug)
        printTimestamp(gpu_id, threadArgs->uuid, "first iter end");

    for (int i = 0; i < 50; ++i)
        NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size, ncclFloat, ncclSum, comm, s));

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));
    if (if_debug)
        printTimestamp(gpu_id, threadArgs->uuid, "end");

    cudaMemcpy(hostData, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU:%d data: %f.\n", gpu_id, hostData[1]);

    ncclCommDestroy(comm);
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    free(hostData);
    return NULL;
}

int main(int argc, char *argv[])
{
    env_init(argc, argv);
    ncclUniqueId id1;
    ncclUniqueId id2;
    CommExec commexec1(my_nranks);
    CommExec commexec2(my_nranks);
    commexec1.launch(id1, 1, allReduceOps);
    commexec2.launch(id2, 2, allReduceOps);
    if (if_debug)
        cmpID(commexec1.get(), commexec2.get());
    commexec2.wait();
    commexec1.wait();
    printf("All streams finished successfully.\n");
    return 0;
}
