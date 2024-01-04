#pragma once
#include <assert.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_bf16.h>
#include <c10/macros/Macros.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>
// #define warpSize    32

// ELEMENTS_PER_LDG = 4, using float2 copy 4 half data. half2 copy 4 uint8_t
template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 1>(c10::BFloat16 *dst, const c10::BFloat16 *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 4>(c10::BFloat16 *dst, const c10::BFloat16 *src) { *((float2*) dst) = *((float2*) src); }

template <>
__device__ __inline__ void copy_vector<c10::Half, 1>(c10::Half *dst, const c10::Half *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<c10::Half, 4>(c10::Half *dst, const c10::Half *src) { *((float2*) dst) = *((float2*) src); }

template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst, const uint8_t *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst, const uint8_t *src) {*((half2*) dst) = *((half2*) src); }

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

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
    return __shfl_xor(value, laneMask, width);
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

// using: DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, function(parameters...))
#define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...)                   \
switch(TYPE)                                                        \
{                                                                   \
case at::ScalarType::Half:                                          \
    {                                                               \
using scalar_t = at::Half;                                          \
__VA_ARGS__;                                                        \
break;                                                              \
    }                                                               \
case at::ScalarType::BFloat16:                                      \
    {                                                               \
using scalar_t = at::BFloat16;                                      \
__VA_ARGS__;                                                        \
break;                                                              \
    }                                                               \
default:                                                            \
    AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");	\
}
