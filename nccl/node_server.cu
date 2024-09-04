
/*
 *  Implement two nodes communication via socket init.
 *  Compile: nvcc -lnccl -ccbin g++ -std=c++11 -O3 -g node_server.cu -o node_server
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
    NCCLCHECK(ncclCommInitRank(&comm, my_nranks*2, id, gpu_id));
    DEBUG_PRINT("============ncclCommInitRank: init end.=============\n"); // debug

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
    DEBUG_PRINT("============ncclAllReduce ===== end =====.\n");

    NCCLCHECK(ncclBroadcast((const void *)recvbuff, (void *)recvbuff, size, ncclFloat, 0, comm, s));
    DEBUG_PRINT("============ncclBroadcast ===== end =====.\n");

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));
    cudaMemcpy(hostData, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU:%d data: %f.\n", gpu_id, hostData[1]);

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
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[1024];

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Cannot create socket" << std::endl;
        return 1;
    }

    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(server_port);

    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Cannot bind" << std::endl;
        return 1;
    }

    if (listen(server_socket, 5) < 0) {
        std::cerr << "Cannot listen" << std::endl;
        return 1;
    }

    std::cout << "Server is listening on port " << server_port << std::endl;

    client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_len);
    if (client_socket < 0) {
        std::cerr << "Cannot accept connection" << std::endl;
        return 1;
    }

    std::cout << "Accepted connection from " << inet_ntoa(client_addr.sin_addr) << std::endl;
    ssize_t recv_size = recv(client_socket, buffer, sizeof(buffer), 0);
    if (recv_size > 0) {
        buffer[recv_size] = '\0';
        std::cout << "Received message: " << buffer << std::endl;
    }

    pthread_t threads[8];

    NCCLCHECK(ncclGetUniqueId(&id));
    if (if_debug)
        std::cout << "=================ncclGetUniqueId================" << buffer << std::endl; // debug

    if (send(client_socket, id.internal, 128, 0) < 0) {
        std::cerr << "Cannot send message to the client" << std::endl;
    }

    close(client_socket);
    close(server_socket);

    for (int i = 0; i < my_nranks; ++i) {
        int *id_pointer = &gpu_ids[i];
        pthread_create(&threads[i], NULL, thread_function, id_pointer);
    }

    for (int i = 0; i < my_nranks; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("Server finished successfully.\n");
    return 0;
}