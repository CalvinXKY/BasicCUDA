/**
 *  warp reduce example.
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


template <typename T> void check(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
                cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor_sync(mask, value, laneMask, width);
    //return __shfl_xor(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

#define WARP_BATCH 1

template<typename data_t>
__global__ void launcher(data_t* src, int nums) {
    data_t tmp[WARP_BATCH] = {0};
    int localIdx= threadIdx.x;
    while (localIdx < nums) {
        tmp[0] += src[localIdx];
        localIdx += gridDim.x * blockDim.x;
    }

    warp_reduce<data_t, WARP_BATCH, 32, Add>(tmp);
    src[threadIdx.x] = tmp[0];
}

int main() {
    unsigned int total_size = 100;
    float* input_data = (float*) malloc(sizeof(float) * total_size);
    float* device_ptr;
    checkCudaErrors(cudaMalloc((void**)&device_ptr, sizeof(float) *total_size));
    for (int i =0; i < 90; ++i) {
        input_data[i] = i * 2;
    }
    checkCudaErrors(cudaMemcpy(device_ptr, input_data,  total_size * sizeof(float), cudaMemcpyHostToDevice));
    launcher<float><<<1, 32>>>(device_ptr, total_size);
    checkCudaErrors(cudaMemcpy(input_data, device_ptr, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Print all data:\n");
    for (int i = 0;i < 10; ++i) {
        for (int k = 0; k < 10 ;++k) {
            printf("%f " ,input_data[i*10 + k]);
        }
        printf("\n");
    }
    checkCudaErrors(cudaFree (device_ptr));
    free(input_data);
    return 0;
}