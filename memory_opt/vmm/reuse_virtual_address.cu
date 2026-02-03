/*
 *  Virtual Memory Management(VMM) example: Reuse Virtual address.
 *
 *  Author: kaiyuan
 *  Email: kaiyuanxie@yeah.net
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>

// 错误检查用的宏函数：
#define CUDA_DRIVER_API_CHECK(call) \
    do { \
        CUresult result = (call); \
        if (result != CUDA_SUCCESS) { \
            const char* errorStr = nullptr; \
            cuGetErrorString(result, &errorStr); \
            std::cerr << "CUDA Driver API error at " << __FILE__ << ":" << __LINE__ \
                      << " - Code: " << result << " - " << (errorStr ? errorStr : "Unknown error") \
                      << " in " << #call << std::endl; \
            exit(1); \
        } \
    } while(0)

const size_t MATRIX_SIZE = 256;
const size_t MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;

// 获取GPU页面大小的辅助函数
size_t getGPUPageSize() {
    // 对于大多数现代GPU，使用2MB页面大小是安全的
    return 2 * 1024 * 1024; // 2MB
}

// 对齐到页面大小
size_t alignToPageSize(size_t size) {
    size_t pageSize = getGPUPageSize();
    return ((size + pageSize - 1) / pageSize) * pageSize;
}

// 简单的矩阵乘法内核
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 启动矩阵乘法的包装函数
void launchMatrixMultiply(CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float* A = reinterpret_cast<float*>(d_A);
    float* B = reinterpret_cast<float*>(d_B);
    float* C = reinterpret_cast<float*>(d_C);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    cudaDeviceSynchronize();
}

// 虚拟内存管理器类
class VirtualMemoryManager {
private:
    CUdeviceptr va_;
    size_t original_size_;
    size_t aligned_size_;
    CUdevice device_;
    CUcontext context_;
    bool is_mapped_;

public:
    VirtualMemoryManager(size_t size) : original_size_(size), is_mapped_(false), context_(nullptr), va_(0) {
        // 计算对齐后的大小：用固定大小对齐。 本例默认size < page size, 会先对齐到page size
        aligned_size_ = alignToPageSize(size);
        // 可以打印出来
        /*
        std::cout << "Original size: " << original_size_
                  << ", Aligned size: " << aligned_size_
                  << ", Page size: " << getGPUPageSize() << std::endl;
         */

        // 初始化CUDA
        CUDA_DRIVER_API_CHECK(cuInit(0));
        CUDA_DRIVER_API_CHECK(cuDeviceGet(&device_, 0));

        // 检查设备能力
        int major = 0, minor = 0;
        CUDA_DRIVER_API_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
        CUDA_DRIVER_API_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
        std::cout << "GPU Compute Capability: " << major << "." << minor << std::endl;

        // 检查虚拟内存管理支持
        int vmm_supported = 0;
        CUresult vmm_result = cuDeviceGetAttribute(&vmm_supported,
                                                  CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                                  device_);
        if (vmm_result == CUDA_SUCCESS) {
            std::cout << "Virtual Memory Management Supported: " << (vmm_supported ? "YES" : "NO") << std::endl;
        }

        // 上下文创建
        std::cout << "Creating context using cuDevicePrimaryCtxRetain..." << std::endl;
        CUDA_DRIVER_API_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));

        // 设置当前上下文
        CUDA_DRIVER_API_CHECK(cuCtxSetCurrent(context_));

        // 预留虚拟地址空间
        CUresult reserve_result = cuMemAddressReserve(&va_, aligned_size_, 0, 0, 0);

        if (reserve_result != CUDA_SUCCESS) {
            // 尝试不同的对齐方式
            std::cout << "First cuMemAddressReserve failed (code: " << reserve_result
                      << "), trying with page size alignment..." << std::endl;
            CUDA_DRIVER_API_CHECK(cuMemAddressReserve(&va_, aligned_size_, getGPUPageSize(), 0, 0));
        }

        std::cout << "Successfully reserved virtual address: 0x" << std::hex << va_ << std::dec
                  << " (size: " << aligned_size_ << " bytes)" << std::endl;
    }

    ~VirtualMemoryManager() {
        if (is_mapped_) {
            unmap();
        }
        if (va_ != 0) {
            CUDA_DRIVER_API_CHECK(cuMemAddressFree(va_, aligned_size_));
            std::cout << "Freed virtual address space" << std::endl;
        }
        if (context_) {
            // 使用正确的释放方法
            cuDevicePrimaryCtxRelease(device_);
        }
    }

    // 映射物理内存到虚拟地址
    CUdeviceptr mapPhysicalMemory(void* host_data = nullptr) {
        if (is_mapped_) {
            std::cerr << "Memory already mapped!" << std::endl;
            return va_;
        }

        CUmemGenericAllocationHandle handle;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;

        // 使用对齐后的大小
        CUDA_DRIVER_API_CHECK(cuMemCreate(&handle, aligned_size_, &prop, 0));
        CUDA_DRIVER_API_CHECK(cuMemMap(va_, aligned_size_, 0, handle, 0));

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = 0;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUDA_DRIVER_API_CHECK(cuMemSetAccess(va_, aligned_size_, &accessDesc, 1));

        if (host_data) {
            // 只复制实际需要的数据
            CUDA_DRIVER_API_CHECK(cuMemcpyHtoD(va_, host_data, original_size_));
        }

        CUDA_DRIVER_API_CHECK(cuMemRelease(handle));

        is_mapped_ = true;
        std::cout << "Physical memory mapped to virtual address 0x" << std::hex << va_ << std::dec << std::endl;
        return va_;
    }

    void unmap() {
        if (!is_mapped_) {
            std::cerr << "No active mapping to unmap!" << std::endl;
            return;
        }

        CUDA_DRIVER_API_CHECK(cuMemUnmap(va_, aligned_size_));
        is_mapped_ = false;
        std::cout << "Unmapped virtual address" << std::endl;
    }

    CUdeviceptr getVirtualAddress() const { return va_; }
    size_t getOriginalSize() const { return original_size_; }
    size_t getAlignedSize() const { return aligned_size_; }
    bool isMapped() const { return is_mapped_; }

    void copyToHost(void* host_buffer) {
        if (!is_mapped_) {
            std::cerr << "No memory mapped!" << std::endl;
            return;
        }
        CUDA_DRIVER_API_CHECK(cuMemcpyDtoH(host_buffer, va_, original_size_));
    }

    void copyFromHost(const void* host_buffer) {
        if (!is_mapped_) {
            std::cerr << "No memory mapped!" << std::endl;
            return;
        }
        CUDA_DRIVER_API_CHECK(cuMemcpyHtoD(va_, host_buffer, original_size_));
    }

    void zeroMemory() {
        if (!is_mapped_) {
            std::cerr << "No memory mapped!" << std::endl;
            return;
        }
        CUDA_DRIVER_API_CHECK(cuMemsetD32(va_, 0, aligned_size_ / sizeof(uint32_t)));
    }
};

// 主程序
int main() {
    const int N = MATRIX_SIZE;
    const size_t matrix_elements = MATRIX_ELEMENTS;
    const size_t matrix_size = matrix_elements * sizeof(float);

    std::cout << "=== Dynamic Virtual Memory Matrix Multiplication Demo ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Elements per matrix: " << matrix_elements << std::endl;
    std::cout << "Original matrix size: " << matrix_size << " bytes" << std::endl;

    // 准备测试数据
    float* host_A1 = new float[matrix_elements];
    float* host_B1 = new float[matrix_elements];
    float* host_C1 = new float[matrix_elements];

    float* host_A2 = new float[matrix_elements];
    float* host_B2 = new float[matrix_elements];
    float* host_C2 = new float[matrix_elements];

    // 初始化数据
    for (size_t i = 0; i < matrix_elements; i++) {
        host_A1[i] = static_cast<float>(i % 10 + 1);
        host_B1[i] = static_cast<float>(i % 5 + 1);
        host_A2[i] = static_cast<float>(i % 7 + 10);
        host_B2[i] = static_cast<float>(i % 3 + 10);
    }

    try {
        // 阶段1：第一个矩阵乘法计算
        std::cout << "\n--- Phase 1: Using first dataset ---" << std::endl;

        VirtualMemoryManager vmm_A(matrix_size);
        VirtualMemoryManager vmm_B(matrix_size);
        VirtualMemoryManager vmm_C(matrix_size);

        CUdeviceptr d_A1 = vmm_A.mapPhysicalMemory(host_A1);
        CUdeviceptr d_B1 = vmm_B.mapPhysicalMemory(host_B1);
        CUdeviceptr d_C1 = vmm_C.mapPhysicalMemory();

        std::cout << "Virtual addresses: A=0x" << std::hex << d_A1
                  << ", B=0x" << d_B1 << ", C=0x" << d_C1 << std::dec << std::endl;

        std::cout << "Executing matrix multiplication (first dataset)..." << std::endl;
        launchMatrixMultiply(d_A1, d_B1, d_C1, N);

        vmm_C.copyToHost(host_C1);
        std::cout << "Phase 1 computation complete" << std::endl;

        // 计算并显示第一阶段的部分结果的求和，默认是5个元素
        float phase1_sum = 0;
        for (size_t i = 0; i < 5; i++) {
            phase1_sum += host_C1[i];
        }
        std::cout << "Phase 1 result sum (first 5 elements): " << phase1_sum << std::endl;

        // 阶段2：第二个矩阵乘法计算
        std::cout << "\n--- Phase 2: Dynamic switch to second dataset ---" << std::endl;


        // 释放物理内存：
        vmm_A.unmap();
        vmm_B.unmap();

        // 旧的虚拟地址重新隐射新物理内存：
        CUdeviceptr d_A2 = vmm_A.mapPhysicalMemory(host_A2);
        CUdeviceptr d_B2 = vmm_B.mapPhysicalMemory(host_B2);

        std::cout << "Virtual addresses comparison:" << std::endl;
        std::cout << "A: 0x" << std::hex << d_A1 << " -> 0x" << d_A2 << std::dec
                  << " (same: " << (d_A1 == d_A2 ? "YES" : "NO") << ")" << std::endl;
        std::cout << "B: 0x" << std::hex << d_B1 << " -> 0x" << d_B2 << std::dec
                  << " (same: " << (d_B1 == d_B2 ? "YES" : "NO") << ")" << std::endl;


        if (d_A1 != d_A2 || d_B1 != d_B2) {
            // 正常情况下是走不到本逻辑
            std::cerr << "ERROR: Virtual addresses changed!" << std::endl;
            return 1;
        }

        std::cout << "Virtual address consistency verified!" << std::endl;

        vmm_C.zeroMemory();

        std::cout << "Executing matrix multiplication (second dataset)..." << std::endl;
        launchMatrixMultiply(d_A2, d_B2, d_C1, N);

        vmm_C.copyToHost(host_C2);
        std::cout << "Phase 2 computation complete" << std::endl;

        // 验证结果
        std::cout << "\n--- Result Verification ---" << std::endl;

        float phase2_sum = 0;
        for (size_t i = 0; i < 5; i++) {
            phase2_sum += host_C2[i];
        }

        std::cout << "Phase 1 result sum (first 5 elements): " << phase1_sum << std::endl;
        std::cout << "Phase 2 result sum (first 5 elements): " << phase2_sum << std::endl;

        // 正常情况下两个结果肯定不同
        if (fabs(phase1_sum - phase2_sum) > 0.001f) {
            std::cout << "Results are different - new data was successfully used!" << std::endl;
        } else {
            std::cout << "Note: Results are similar, but virtual address reuse was demonstrated." << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    // 清理
    delete[] host_A1;
    delete[] host_B1;
    delete[] host_C1;
    delete[] host_A2;
    delete[] host_B2;
    delete[] host_C2;

    std::cout << "\n --- Demo completed successfully! --- " << std::endl;
    return 0;
}
