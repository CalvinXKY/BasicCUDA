

#include "matMul.h"

template <int BLOCK_SIZE> __global__ void MatMulKernel2DAnySize(float *C, float *A, float *B, int wA, int wC, int hC)
{
    int wB = wC;
    int maxIdxA = wA * hC;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    while (wA * BLOCK_SIZE * by < maxIdxA) {
        // Index of the first sub-matrix of A processed by the block
        int aBegin = wA * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd = aBegin + wA - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep = BLOCK_SIZE * wB;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        float Csub = 0;

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        int flag = 0;
        for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            if (flag * BLOCK_SIZE + tx < wA || flag * BLOCK_SIZE + ty < hC) {
                As[ty][tx] = A[a + wA * ty + tx];
            } else {
                As[ty][tx] = 0.0;
            }

            if (flag * BLOCK_SIZE + ty < wA || flag * BLOCK_SIZE + tx < wC) {
                Bs[ty][tx] = B[b + wB * ty + tx];
            } else {
                Bs[ty][tx] = 0.0;
            }

            // Bs[ty][tx] = B[idx];

            // Bs[ty][tx] = B[idx];
            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
#pragma unroll

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                Csub += As[ty][k] * Bs[k][tx];
            }

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
            flag++;
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        if (BLOCK_SIZE * bx + tx < wC && BLOCK_SIZE * by + ty < hC) { // thread could over max.
            C[wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx] = Csub;
        }
        bx += BLOCK_SIZE;
        by += BLOCK_SIZE;
    }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatMulKernel2DBlockMultiplesSize(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

/**
 * Run a simple test of matrix multiplication with 2D blocks.
 */
int MatMul2DTest(int argc, char **argv, int blockSize, int iterNum, const dim3 &dimsA, const dim3 &dimsB,
                 bool useAnySize)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    cudaStream_t stream;

    // Initialize host memory
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(blockSize, blockSize);
    dim3 grid;

    // Create and start timer
    printf("Computing result using MatMul2DTest Kernel. \n");
    if (useAnySize)
        printf("Spport any size, e.g. wA=1000 hA=312 wB=11 hB=1000.\n");

    // select diff blocks for kerenl
    void (*MMKernel2DAnySizeExe)(float *, float *, float *, int, int, int);
    void (*MMKernel2DFixSizeExe)(float *, float *, float *, int, int);
    if (blockSize <= 16) {
        MMKernel2DFixSizeExe = MatMulKernel2DBlockMultiplesSize<16>;
        MMKernel2DAnySizeExe = MatMulKernel2DAnySize<16>;
    } else {
        MMKernel2DFixSizeExe = MatMulKernel2DBlockMultiplesSize<32>;
        MMKernel2DAnySizeExe = MatMulKernel2DAnySize<32>;
    }

    // Performs warmup operation using matrixMul CUDA kernel
    if (useAnySize) {
        grid = dim3(dimsB.x / threads.x + 1, dimsA.y / threads.y + 1);
        MMKernel2DAnySizeExe<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);
    } else {
        grid = dim3(dimsB.x / threads.x, dimsA.y / threads.y);
        MMKernel2DFixSizeExe<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("Warmup  operation done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    for (int j = 0; j < iterNum; j++) {
        if (useAnySize) {
            MMKernel2DAnySizeExe<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);
        } else {
            MMKernel2DFixSizeExe<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / iterNum;
    double flopsPerMatrixMul =
        2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    bool ret = ResultCheck(h_C, static_cast<int>(dimsC.x * dimsC.y), dimsA.x, valB);

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));

    if (ret) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
