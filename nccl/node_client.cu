/*
 *  Implement two nodes communication via socket init.
 *  Compile: nvcc -lnccl -ccbin g++ -std=c++11 -O3 -g node_client.cu -o node_client
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "comm.h"

ncclUniqueId id;

void *thread_function(void *arg)
{
    int size = 32 * 1024;
    int gpu_id = *(int *)arg;
    cudaSetDevice(gpu_id);

    ncclComm_t comm;
    if (if_debug)
        std::cout << "Received from server: " << id.internal << std::endl; // debug

    NCCLCHECK(ncclCommInitRank(&comm, my_nranks*2, id, gpu_id + my_nranks));

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
    DEBUG_PRINT("============ncclAllReduce == end=====.\n");
    NCCLCHECK(ncclBroadcast((const void *)sendbuff, (void *)recvbuff, size, ncclFloat, 0, comm, s));
    DEBUG_PRINT("============ncclBroadcast == end=====.\n");

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));
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
    int sock;
    struct sockaddr_in server_addr;
    const char *message = "Hello, server!";

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Cannot create socket" << std::endl;
        return 1;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    inet_pton(AF_INET, server_hostname.c_str(), &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Cannot connect to the server" << std::endl;
        return 1;
    }

    std::cout << "Connected to the server" << std::endl;

    if (send(sock, message, strlen(message), 0) < 0) {
        std::cerr << "Cannot send message" << std::endl;
        return 1;
    }

    ssize_t recv_size = recv(sock, id.internal, 128, 0);
    if (recv_size > 0 && if_debug) {
        std::cout << "Received from server: " << id.internal << std::endl;
    }
    close(sock);

    pthread_t threads[8];
    for (int i = 0; i < my_nranks; ++i) {
        int *id_pointer = &gpu_ids[i];
        pthread_create(&threads[i], NULL, thread_function, id_pointer);
    }

    for (int i = 0; i < my_nranks; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads finished successfully.\n");

    return 0;
}