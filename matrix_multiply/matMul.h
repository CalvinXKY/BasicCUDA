#pragma once
// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_profiler_api.h>
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
                result, func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

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

inline void ConstantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

inline bool ResultCheck(float *h_C,int sizeC, int wA, const float valB) {
    printf("Checking computed result for correctness: ");
    bool correct = true;
    
    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6; // machine zero

    for (int i = 0; i < sizeC; i++) {
        double abs_err = fabs(h_C[i] - (wA* valB));
        double dot_length = wA;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], wA * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
    return correct;
}

int MatrixMul1DTest(int argc, char **argv, int threadSize, int iterNum, const dim3 &dimsA, const dim3 &dimsB,
                    bool useShMem);

int MatMul2DTest(int argc, char **argv, int thblockSize, int iterNum, const dim3 &dimsA, const dim3 &dimsB,
                 bool useAnySize);

int MatMulCublasTest(int argc, char **argv, int blockSize, int iterNum, const dim3 &dimsA, const dim3 &dimsB);
