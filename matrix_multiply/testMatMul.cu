/**
 * Test different version of matrix multiply.
 *   A x B   A[hA, wA] B[hB, wB]
 * e.g. ./matMul wA=1000 hA=312 wB=11 hB=1000
 *
 * Author: kevin.xie
 * Email: kaiyuanxie@yeah.net
 */

#include "matMul.h"

enum ALGO_TYPE {
    DEFAULT_MODEL,
    MatMul_1D_KERENL,
    MatMul_1D_KERNEL_WITH_SHARED_MEMORY,
    MatMul_2D_KERENEL_BLOCK_MULTIPLES_SIZE,
    MatMul_2D_KERNEL_ANY_SIZE,
};

/**
 * Program main
 */

void checkResult(int ret)
{
    if (ret != EXIT_SUCCESS) {
        checkCudaErrors(cudaProfilerStop());
        exit(ret);
    }
}

int main(int argc, char **argv)
{
    printf("[Matrix Multiply Test] - Starting...\n");
    printf("\nNOTE: The CUDA Samples are not meant for performance "
           "measurements. Results may vary when GPU Boost is enabled.\n");
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("      -iter=n Iteration numbers of algorithm. Default:500 \n");
        printf("      -algo=[0|1|2|3|4] 0: Test all, 1: MatMul_1D_KERENL, 2:MatMul_1D_KERNEL_WITH_SHARED_MEMORY, "
               "3: MatMul_2D_KERENEL_BLOCK_MULTIPLES_SIZE, 4: MatMul_2D_KERNEL_ANY_SIZE\n");
        printf("Note: Outer matrix dimensions of A & B matrices"
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // int dev = 0;
    int blockSize = 32;
    int threadsPerBlock = blockSize * blockSize;

    // select algorithem:
    int algo = 0;
    int iterationNum = 500;

    // example case:
    dim3 dimsA(5 * 2 * blockSize, 5 * 2 * blockSize, 1);
    dim3 dimsB(5 * 4 * blockSize, 5 * 2 * blockSize, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iter")) {
        iterationNum = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "algo")) {
        algo = getCmdLineArgumentInt(argc, (const char **)argv, "algo");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    // int matrix_result = MatrixMul1DTest(argc, argv, 256, iterationNum, dimsA, dimsB, false);
    checkCudaErrors(cudaProfilerStart());
    switch (algo) {
        case MatMul_1D_KERENL:
            checkResult(MatrixMul1DTest(argc, argv, threadsPerBlock, iterationNum, dimsA, dimsB, false));
            break;
        case MatMul_1D_KERNEL_WITH_SHARED_MEMORY:
            checkResult(MatrixMul1DTest(argc, argv, threadsPerBlock, iterationNum, dimsA, dimsB, true));
            break;
        case MatMul_2D_KERENEL_BLOCK_MULTIPLES_SIZE:
            if (dimsA.x % blockSize != 0) {
                printf("dim of wA must be divided by blockSize: %d\n", blockSize);
                exit(EXIT_FAILURE);
            }
            checkResult(MatMul2DTest(argc, argv, blockSize, iterationNum, dimsA, dimsB, false));
            break;
        case MatMul_2D_KERNEL_ANY_SIZE:
            checkResult(MatMul2DTest(argc, argv, blockSize, iterationNum, dimsA, dimsB, true));
            break;
        default:
            printf("========================= 1D blocks without shared memory =================\n");
            checkResult(MatrixMul1DTest(argc, argv, threadsPerBlock, iterationNum, dimsA, dimsB, false));
            printf("========================= 1D blocks with shared memory ===================\n");
            checkResult(MatrixMul1DTest(argc, argv, threadsPerBlock, iterationNum, dimsA, dimsB, true));
            if (dimsA.x % blockSize == 0) {
                printf("========================= 2D blocks with block multiples size =============\n");
                checkResult(MatMul2DTest(argc, argv, blockSize, iterationNum, dimsA, dimsB, false));
            }
            printf("========================= 2D blocks with any size ========================\n");
            checkResult(MatMul2DTest(argc, argv, blockSize, iterationNum, dimsA, dimsB, true));
            break;
    }

    checkCudaErrors(cudaProfilerStop());
    exit(EXIT_SUCCESS);
}
