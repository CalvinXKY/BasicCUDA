#pragma once
// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define STRCASECMP _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP strcasecmp
#define STRNCASECMP strncasecmp
#endif

template <typename T> void check(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define CACHE_CLEAR_SIZE (16 * (1e6)) // 16 M

#define TIME_ELAPSE(func, elapsedTime, start, stop)  \
    cudaEventCreate(&start);                         \
    cudaEventCreate(&stop);                          \
    cudaEventRecord(start, 0);                       \
    (func);                                          \
    cudaEventRecord(stop, 0);                        \
    cudaEventSynchronize(stop);                      \
    cudaEventElapsedTime(&elapsedTime, start, stop); \
    cudaEventDestroy(start);                         \
    cudaEventDestroy(stop);

inline int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;

    while (string[string_start] == delimiter) {
        string_start++;
    }

    if (string_start >= static_cast<int>(strlen(string) - 1)) {
        return 0;
    }

    return string_start;
}

inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, '=');
            int argv_length = static_cast<int>(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = static_cast<int>(strlen(string_ref));

            if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length)) {
                bFound = true;
                continue;
            }
        }
    }

    return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;
    int value = -1;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = static_cast<int>(strlen(string_ref));

            if (!STRNCASECMP(string_argv, string_ref, length)) {
                if (length + 1 <= static_cast<int>(strlen(string_argv))) {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
                } else {
                    value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound) {
        return value;
    } else {
        return 0;
    }
}
