/*
 *  Virtual Memory Management(VMM) example: Work with cuda graph.
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
#include <chrono>


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

#define CUDA_RUNTIME_CHECK(call) \
    do { \
        cudaError_t result = (call); \
        if (result != cudaSuccess) { \
            std::cerr << "CUDA Runtime error at " << __FILE__ << ":" << __LINE__ \
                      << " - Code: " << result << " - " << cudaGetErrorString(result) \
                      << " in " << #call << std::endl; \
            exit(1); \
        } \
    } while(0)

const size_t MATRIX_SIZE = 256;
const size_t MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;

// 获取GPU页面大小的辅助函数
size_t getGPUPageSize() {
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
void launchMatrixMultiply(CUdeviceptr d_A, CUdeviceptr d_B, CUdeviceptr d_C, int N, cudaStream_t stream = 0) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float* A = reinterpret_cast<float*>(d_A);
    float* B = reinterpret_cast<float*>(d_B);
    float* C = reinterpret_cast<float*>(d_C);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(A, B, C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
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
        aligned_size_ = alignToPageSize(size);

        // 初始化CUDA
        CUDA_DRIVER_API_CHECK(cuInit(0));
        CUDA_DRIVER_API_CHECK(cuDeviceGet(&device_, 0));

        // 使用cuDevicePrimaryCtxRetain
        CUDA_DRIVER_API_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
        CUDA_DRIVER_API_CHECK(cuCtxSetCurrent(context_));

        // 预留虚拟地址空间
        CUresult reserve_result = cuMemAddressReserve(&va_, aligned_size_, 0, 0, 0);

        if (reserve_result != CUDA_SUCCESS) {
            CUDA_DRIVER_API_CHECK(cuMemAddressReserve(&va_, aligned_size_, getGPUPageSize(), 0, 0));
        }

        std::cout << "Reserved VA: 0x" << std::hex << va_ << std::dec
                  << " (aligned: " << aligned_size_ << " bytes)" << std::endl;
    }

    ~VirtualMemoryManager() {
        if (is_mapped_) unmap();
        if (va_ != 0) {
            CUDA_DRIVER_API_CHECK(cuMemAddressFree(va_, aligned_size_));
        }
        if (context_) {
            cuDevicePrimaryCtxRelease(device_);
        }
    }

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

        CUDA_DRIVER_API_CHECK(cuMemCreate(&handle, aligned_size_, &prop, 0));
        CUDA_DRIVER_API_CHECK(cuMemMap(va_, aligned_size_, 0, handle, 0));

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = 0;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUDA_DRIVER_API_CHECK(cuMemSetAccess(va_, aligned_size_, &accessDesc, 1));

        if (host_data) {
            CUDA_DRIVER_API_CHECK(cuMemcpyHtoD(va_, host_data, original_size_));
        }

        CUDA_DRIVER_API_CHECK(cuMemRelease(handle));

        is_mapped_ = true;
        std::cout << "Mapped physical memory to VA: 0x" << std::hex << va_ << std::dec << std::endl;
        return va_;
    }

    void unmap() {
        if (!is_mapped_) return;
        CUDA_DRIVER_API_CHECK(cuMemUnmap(va_, aligned_size_));
        is_mapped_ = false;
    }

    CUdeviceptr getVirtualAddress() const { return va_; }
    bool isMapped() const { return is_mapped_; }

    void copyToHost(void* host_buffer) {
        if (!is_mapped_) return;
        CUDA_DRIVER_API_CHECK(cuMemcpyDtoH(host_buffer, va_, original_size_));
    }

    void zeroMemory() {
        if (!is_mapped_) return;
        CUDA_DRIVER_API_CHECK(cuMemsetD32(va_, 0, aligned_size_ / sizeof(uint32_t)));
    }
};

// 比较两个数组是否相等
bool compareArrays(const float* a, const float* b, size_t n, float epsilon = 1e-6) {
    for (size_t i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 辅助函数：将cudaGraphExecUpdateResult转换为字符串
const char* getUpdateResultString(cudaGraphExecUpdateResult result) {
    switch (result) {
        case cudaGraphExecUpdateSuccess: return "Success";
        case cudaGraphExecUpdateError: return "Error";
        case cudaGraphExecUpdateErrorTopologyChanged: return "TopologyChanged";
        case cudaGraphExecUpdateErrorNodeTypeChanged: return "NodeTypeChanged";
        case cudaGraphExecUpdateErrorAttributesChanged: return "AttributesChanged";
        case cudaGraphExecUpdateErrorParametersChanged: return "ParametersChanged";
        case cudaGraphExecUpdateErrorNotSupported: return "NotSupported";
        default: return "Unknown";
    }
}

int main() {
    const int N = MATRIX_SIZE;
    const size_t matrix_elements = MATRIX_ELEMENTS;
    const size_t matrix_size = matrix_elements * sizeof(float);

    std::cout << "=== CUDA Graph with Virtual Memory and Update Check ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Matrix size: " << matrix_size / (1024*1024) << " MB" << std::endl;

    // 准备测试数据
    float* host_A1 = new float[matrix_elements];
    float* host_B1 = new float[matrix_elements];
    float* host_C1 = new float[matrix_elements];
    float* host_C1_direct = new float[matrix_elements];

    float* host_A2 = new float[matrix_elements];
    float* host_B2 = new float[matrix_elements];
    float* host_C2 = new float[matrix_elements];
    float* host_C2_direct = new float[matrix_elements];

    // 初始化数据
    for (size_t i = 0; i < matrix_elements; i++) {
        host_A1[i] = static_cast<float>(i % 10 + 1);
        host_B1[i] = static_cast<float>(i % 5 + 1);
        host_A2[i] = static_cast<float>(i % 7 + 10);
        host_B2[i] = static_cast<float>(i % 3 + 10);
    }

    // CUDA Graph相关变量
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t stream;

    try {
        // ========== 阶段1: 初始化和首次Graph捕获 ==========
        std::cout << "\n--- Phase 1: Initial Graph Capture ---" << std::endl;

        VirtualMemoryManager vmm_A(matrix_size);
        VirtualMemoryManager vmm_B(matrix_size);
        VirtualMemoryManager vmm_C(matrix_size);

        CUdeviceptr d_A1 = vmm_A.mapPhysicalMemory(host_A1);
        CUdeviceptr d_B1 = vmm_B.mapPhysicalMemory(host_B1);
        CUdeviceptr d_C1 = vmm_C.mapPhysicalMemory();

        std::cout << "Virtual addresses:\n";
        std::cout << "  A: 0x" << std::hex << d_A1 << std::dec << "\n";
        std::cout << "  B: 0x" << std::hex << d_B1 << std::dec << "\n";
        std::cout << "  C: 0x" << std::hex << d_C1 << std::dec << std::endl;

        // 创建流
        CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

        // 首次Graph捕获
        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "Capturing CUDA Graph..." << std::endl;
        CUDA_RUNTIME_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launchMatrixMultiply(d_A1, d_B1, d_C1, N, stream);
        CUDA_RUNTIME_CHECK(cudaStreamEndCapture(stream, &graph));

        // 实例化Graph
        CUDA_RUNTIME_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Graph capture+instantiate time: " << duration.count() << "us" << std::endl;

        // 执行Graph
        std::cout << "Executing captured Graph..." << std::endl;
        CUDA_RUNTIME_CHECK(cudaGraphLaunch(graph_exec, stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        vmm_C.copyToHost(host_C1);

        // 直接执行作为参考
        vmm_C.zeroMemory();
        launchMatrixMultiply(d_A1, d_B1, d_C1, N);
        cudaDeviceSynchronize();
        vmm_C.copyToHost(host_C1_direct);

        if (compareArrays(host_C1, host_C1_direct, 100)) {
            std::cout << "Phase 1: Graph execution matches direct execution" << std::endl;
        }

        // ========== 阶段2: 数据切换和Graph更新检查 ==========
        std::cout << "\n--- Phase 2: Data Switching and Graph Update Check ---" << std::endl;

        // 步骤1: 卸载旧数据，加载新数据到相同虚拟地址
        vmm_A.unmap();
        vmm_B.unmap();

        std::cout << "Switching datasets while keeping virtual addresses unchanged..." << std::endl;
        CUdeviceptr d_A2 = vmm_A.mapPhysicalMemory(host_A2);
        CUdeviceptr d_B2 = vmm_B.mapPhysicalMemory(host_B2);

        // 验证虚拟地址确实未变
        std::cout << "Virtual address verification:\n";
        std::cout << "  A: 0x" << std::hex << d_A1 << " -> 0x" << d_A2
                  << " (same: " << (d_A1 == d_A2 ? "YES" : "NO") << ")\n";
        std::cout << "  B: 0x" << std::hex << d_B1 << " -> 0x" << d_B2
                  << " (same: " << (d_B1 == d_B2 ? "YES" : "NO") << ")" << std::dec << std::endl;

        if (d_A1 != d_A2 || d_B1 != d_B2) {
            std::cerr << "ERROR: Virtual addresses changed!" << std::endl;
            return 1;
        }

        // 步骤2: 关键 - 捕获一个新的临时Graph用于更新检查
        std::cout << "\nCapturing new graph with updated data..." << std::endl;
        cudaGraph_t new_graph;
        CUDA_RUNTIME_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launchMatrixMultiply(d_A2, d_B2, d_C1, N, stream);
        CUDA_RUNTIME_CHECK(cudaStreamEndCapture(stream, &new_graph));
        std::cout << "New graph captured successfully" << std::endl;

        // 步骤3: 使用cudaGraphExecUpdate检查Graph兼容性
        std::cout << "\nChecking if existing graph_exec can be updated with new graph..." << std::endl;
        cudaGraphExecUpdateResult update_result;
        cudaGraphNode_t error_node = nullptr;

        start_time = std::chrono::high_resolution_clock::now();
        cudaError_t update_status = cudaGraphExecUpdate(graph_exec, new_graph, &error_node, &update_result);
        end_time = std::chrono::high_resolution_clock::now();

        auto update_check_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Update check completed in " << update_check_time.count() << "us" << std::endl;
        std::cout << "Update result: " << getUpdateResultString(update_result)
                  << " (code: " << update_result << ")" << std::endl;

        // 步骤4: 根据更新结果采取不同行动
        if (update_status == cudaSuccess && update_result == cudaGraphExecUpdateSuccess) {
            std::cout << "\ncudaGraphExecUpdate SUCCESS!" << std::endl;
            std::cout << "Meaning: The existing graph_exec can handle the new data without recompilation." << std::endl;
            std::cout << "Reason: Only memory content changed, addresses remain identical." << std::endl;

            // 无需重新实例化，直接重用现有graph_exec
            std::cout << "\nReusing existing graph_exec with new data..." << std::endl;
            vmm_C.zeroMemory();

            start_time = std::chrono::high_resolution_clock::now();
            CUDA_RUNTIME_CHECK(cudaGraphLaunch(graph_exec, stream));
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
            end_time = std::chrono::high_resolution_clock::now();

            auto reuse_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Graph reuse execution time: " << reuse_time.count() << "us" << std::endl;

            // 获取结果
            vmm_C.copyToHost(host_C2);

        } else {
            std::cout << "\ncudaGraphExecUpdate FAILED!" << std::endl;
            std::cout << "Meaning: The existing graph_exec cannot handle the new data." << std::endl;
            std::cout << "Action: Need to destroy old graph_exec and instantiate a new one." << std::endl;

            // 需要重新实例化
            CUDA_RUNTIME_CHECK(cudaGraphExecDestroy(graph_exec));
            graph_exec = nullptr;

            std::cout << "Instantiating new graph_exec..." << std::endl;
            start_time = std::chrono::high_resolution_clock::now();
            CUDA_RUNTIME_CHECK(cudaGraphInstantiate(&graph_exec, new_graph, nullptr, nullptr, 0));
            end_time = std::chrono::high_resolution_clock::now();

            auto recompile_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Recompilation time: " << recompile_time.count() << "us" << std::endl;

            // 执行重新编译后的graph
            vmm_C.zeroMemory();
            CUDA_RUNTIME_CHECK(cudaGraphLaunch(graph_exec, stream));
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
            vmm_C.copyToHost(host_C2);
        }

        // 步骤5: 验证新结果正确性
        std::cout << "\nVerifying results with new data..." << std::endl;

        // 直接执行作为比较基准
        vmm_C.zeroMemory();
        launchMatrixMultiply(d_A2, d_B2, d_C1, N);
        cudaDeviceSynchronize();
        vmm_C.copyToHost(host_C2_direct);

        if (compareArrays(host_C2, host_C2_direct, 100)) {
            std::cout << "Phase 2: Results are correct" << std::endl;
        }

        // 步骤6: 对比新旧结果，证明数据确实变了
        float sum1 = 0, sum2 = 0;
        for (size_t i = 0; i < 10; i++) {
            sum1 += host_C1[i];
            sum2 += host_C2[i];
        }
        std::cout << "\nData change verification:" << std::endl;
        std::cout << "  Phase 1 result sum (first 10): " << sum1 << std::endl;
        std::cout << "  Phase 2 result sum (first 10): " << sum2 << std::endl;
        std::cout << "  Difference: " << fabs(sum1 - sum2) << std::endl;

        if (fabs(sum1 - sum2) > 0.001f) {
            std::cout << "Confirmed: Computation used new dataset" << std::endl;
        }

        // 清理临时graph
        CUDA_RUNTIME_CHECK(cudaGraphDestroy(new_graph));

        std::cout << "Graph successfully reused: "
                  << (update_result == cudaGraphExecUpdateSuccess ? "YES" : "NO") << std::endl;

        // 清理CUDA资源
        if (graph_exec) CUDA_RUNTIME_CHECK(cudaGraphExecDestroy(graph_exec));
        if (graph) CUDA_RUNTIME_CHECK(cudaGraphDestroy(graph));
        CUDA_RUNTIME_CHECK(cudaStreamDestroy(stream));

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    // 清理主机内存
    delete[] host_A1;
    delete[] host_B1;
    delete[] host_C1;
    delete[] host_C1_direct;
    delete[] host_A2;
    delete[] host_B2;
    delete[] host_C2;
    delete[] host_C2_direct;

    std::cout << "\nDemo completed successfully!" << std::endl;
    return 0;
}