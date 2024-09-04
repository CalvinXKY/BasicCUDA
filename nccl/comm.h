#pragma once

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>

#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd)                                                                              \
    do {                                                                                            \
        cudaError_t err = cmd;                                                                      \
        if (err != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

#define NCCLCHECK(cmd)                                                                              \
    do {                                                                                            \
        ncclResult_t res = cmd;                                                                     \
        if (res != ncclSuccess) {                                                                   \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

#define DEFAULT_DEVICES_NUM 8
int gpu_ids[8] = {0, 1, 2, 3, 4, 5, 6, 7}; 
char *if_debug = nullptr;
std::string server_hostname = "127.0.0.1";
int server_port = 8099;
int my_nranks = 6;

#define DEBUG_PRINT(info)                       \
    if (if_debug && strcasecmp(if_debug, "0") != 0) {    \
        printf("DEUBG INFO: %s\n", info); \
    }

const std::string help_info = "Usage: --nranks     The number of ranks/GPU\n\
        --hostname   Server IP address.\n\
        --port       To specify a port. Default: 8099 \n\
E.g. ./run --nranks 4 --port 8096\n";

void env_init(int argc, char* argv[])
{
    if_debug = getenv("DEBUG");
    std::map<std::string, std::string> options;
    const std::set<std::string> allow_options{"--nranks", "--hostname", "--port"};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.substr(0, 2) == "--") {
            std::string value;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                value = argv[i + 1];
                i++;
            }
            options[arg] = value;
        } else {
            std::cout << "Unknown option: " << arg << std::endl;
            std::cout << help_info << std::endl;
            exit(-1);
        }
    }

    for (const auto &opt : options) {
        if (allow_options.find(opt.first) == allow_options.end()) {
            std::cout << "Unknown option: " << opt.first << std::endl << help_info;
            exit(-1);
        }
    }

    if (options.find("--nranks") != options.end()) {
        std::cout << "Local rank size: " << options["--nranks"] << std::endl;
        my_nranks = std::stoi(options["--nranks"]);
    }
    if (options.find("--hostname") != options.end()) {
        std::cout << "The hostname: " << options["--hostname"] << std::endl;
        server_hostname = options["--hostname"];
    }
    if (options.find("--port") != options.end()) {
        std::cout << "The hostport: " << options["--port"] << std::endl;
        server_port = std::stoi(options["--port"]);
    }
}
