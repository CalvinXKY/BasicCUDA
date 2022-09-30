
#include "matMul.h"

__global__ void MatMulKernel1D(float *C, float *A, float *B, const int wh, const int wC, const int hC)
{
    const int totalSize = wC * hC;
    int thID = threadIdx.x + blockIdx.x * blockDim.x;
    while (thID < totalSize) {
        int Cx = thID / wC;
        int Cy = thID % wC;
        float rst = 0.0;
        for (int i = 0; i < wh; i++) {
            rst += A[Cx * wh + i] * B[i * wC + Cy];
        }
        C[Cx * wC + Cy] = rst;
        thID += gridDim.x * blockDim.x;
    }
    __syncthreads();
}

template <int shWASize>
__global__ void MatMulKernel1DWithShMem(float *C, float *A, float *B, const int wA, const int wC, const int hC)
{
    __shared__ float sRow[shWASize]; // shared wA
    int blockID = blockIdx.x;
    while (blockID < hC) {
        int thIdx = threadIdx.x;
        while (thIdx < wA) {
            sRow[thIdx] = A[blockID * wA + thIdx];
            thIdx += blockDim.x;
        }
        __syncthreads();

        thIdx = threadIdx.x;
        while (thIdx < wC) { // wB = wC;
            float sum = 0.0;
            for (int i = 0; i < wA; i++) {
                sum += sRow[i] * B[wC * i + thIdx];
            }
            C[blockID * wC + thIdx] = sum;
            thIdx += blockDim.x;
        }
        blockID += gridDim.x;
    }
}


/*
* Run a simple test of matrix multiplication with 1D blocks.
*/
int MatrixMul1DTest(int argc, char **argv, int threadSize, int iterNum, const dim3 &dimsA, const dim3 &dimsB,
                    bool useShMem)
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
    int grid = dimsC.x * dimsC.y / threadSize;
    // dim3 grid(4, 4);

    // Create and start timer
    printf("Computing result using MatrixMul1DTest Shared Mem: %d\n", useShMem);

    // select diff shared memory size in blocks;
    void (*MMKernel1DWithShMemExe)(float *C, float *A, float *B, const int wA, const int wC, const int hC);
    if (dimsA.x <= 256) {
        MMKernel1DWithShMemExe = MatMulKernel1DWithShMem<256>;
    } else if (dimsA.x <= 1024) {
        MMKernel1DWithShMemExe = MatMulKernel1DWithShMem<1024>;
    } else if (dimsA.x <= 2048) {
        MMKernel1DWithShMemExe = MatMulKernel1DWithShMem<2048>;
    } else if (dimsA.x <= 4096) {
        MMKernel1DWithShMemExe = MatMulKernel1DWithShMem<4096>;
    } else {
        // shared mem has limitation. Change the size according to your scenarios.
        MMKernel1DWithShMemExe = MatMulKernel1DWithShMem<8192>;
    }

    // Performs warmup operation using matrixMul CUDA kernel
    if (useShMem) {
        MMKernel1DWithShMemExe<<<grid, threadSize, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);
    } else {
        MatMulKernel1D<<<grid, threadSize, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);
    }
    printf("Warmup  operation done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    for (int j = 0; j < iterNum; j++) {
        if (useShMem) {
            MMKernel1DWithShMemExe<<<grid, threadSize, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);
        } else {
            MatMulKernel1D<<<grid, threadSize, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsC.x, dimsC.y);
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
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threadSize);

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
