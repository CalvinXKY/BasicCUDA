
# Virtual Memory Management(VMM)虚拟地址

## Case1：虚拟地址基本使用

本例目的：
* 掌握虚拟地址的基本使用；
* 验证虚拟地址可以重复使用；

基本思路：
- 应用虚拟地址创建数据，并运算矩阵乘法；
- 一共有两个矩阵乘法，依次运算；
- 第一个矩阵乘法运算完后释放物理显存，但不释放虚拟地址继续用于第二个矩阵乘法。
- 第二个矩阵运算完后，比较两次的地址与结果。

代码：reuse_virtual_address.cu

编译方式：nvcc -o reuse_virtual_address reuse_virtual_address.cu -lcuda

运行方式：./reuse_virtual_address

测试运行环境：
```
# nvcc --verison

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_Dec_16_07:23:41_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.115
Build cuda_13.1.r13.1/compiler.37061995_0

# nvidia-smi
NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 13.1

```
建议使用nvidia官方镜像运行：[Link](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=pytorch)

镜像：nvcr.io/nvidia/pytorch:xx.xx-py3

输出示例:

```
=== Dynamic Virtual Memory Matrix Multiplication Demo ===
Matrix size: 256x256
Elements per matrix: 65536
Original matrix size: 262144 bytes

--- Phase 1: Using first dataset ---
GPU Compute Capability: 8.0
Virtual Memory Management Supported: YES
Creating context using cuDevicePrimaryCtxRetain...
Successfully reserved virtual address: 0x7f7ff1200000 (size: 2097152 bytes)
GPU Compute Capability: 8.0
Virtual Memory Management Supported: YES
Creating context using cuDevicePrimaryCtxRetain...
Successfully reserved virtual address: 0x7f7ff1400000 (size: 2097152 bytes)
GPU Compute Capability: 8.0
Virtual Memory Management Supported: YES
Creating context using cuDevicePrimaryCtxRetain...
Successfully reserved virtual address: 0x7f7ff1600000 (size: 2097152 bytes)
Physical memory mapped to virtual address 0x7f7ff1200000
Physical memory mapped to virtual address 0x7f7ff1400000
Physical memory mapped to virtual address 0x7f7ff1600000
Virtual addresses: A=0x7f7ff1200000, B=0x7f7ff1400000, C=0x7f7ff1600000
Executing matrix multiplication (first dataset)...
Phase 1 computation complete
Phase 1 result sum (first 5 elements): 20940

--- Phase 2: Dynamic switch to second dataset ---
Unmapped virtual address
Unmapped virtual address
Physical memory mapped to virtual address 0x7f7ff1200000
Physical memory mapped to virtual address 0x7f7ff1400000
Virtual addresses comparison:
A: 0x7f7ff1200000 -> 0x7f7ff1200000 (same: YES)
B: 0x7f7ff1400000 -> 0x7f7ff1400000 (same: YES)
Virtual address consistency verified!
Executing matrix multiplication (second dataset)...
Phase 2 computation complete

--- Result Verification ---
Phase 1 result sum (first 5 elements): 20940
Phase 2 result sum (first 5 elements): 182698
Results are different - new data was successfully used!
Unmapped virtual address
Freed virtual address space
Unmapped virtual address
Freed virtual address space
Unmapped virtual address
Freed virtual address space

 --- Demo completed successfully! ---

```

## Case2：虚拟地址与CUDA Graph

本例目的：
* 演示虚拟地址与CUDA Graph配合使用；
* 虚拟地址保持不变，不会触发CUDA Graph的重新编译。

测试中验证Graph是否需要重编译，函数：cudaGraphExecUpdate

virtual_mem_with_cuda_graph.cu

代码：virtual_mem_with_cuda_graph.cu

编译方式：nvcc -o virtual_mem_with_cuda_graph virtual_mem_with_cuda_graph.cu -lcuda

运行方式： ./virtual_mem_with_cuda_graph

测试运行环境：
```
# nvcc --verison

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_Dec_16_07:23:41_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.115
Build cuda_13.1.r13.1/compiler.37061995_0

# nvidia-smi
NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 13.1

```
建议使用nvidia官方镜像运行：[Link](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=pytorch)

镜像：nvcr.io/nvidia/pytorch:xx.xx-py3

输出示例:

```
=== CUDA Graph with Virtual Memory and Update Check ===
Matrix size: 256x256
Matrix size: 0 MB

--- Phase 1: Initial Graph Capture ---
Reserved VA: 0x7f5a3d200000 (aligned: 2097152 bytes)
Reserved VA: 0x7f5a3d400000 (aligned: 2097152 bytes)
Reserved VA: 0x7f5a3d600000 (aligned: 2097152 bytes)
Mapped physical memory to VA: 0x7f5a3d200000
Mapped physical memory to VA: 0x7f5a3d400000
Mapped physical memory to VA: 0x7f5a3d600000
Virtual addresses:
  A: 0x7f5a3d200000
  B: 0x7f5a3d400000
  C: 0x7f5a3d600000
Capturing CUDA Graph...
Graph capture+instantiate time: 482us
Executing captured Graph...
Phase 1: Graph execution matches direct execution

--- Phase 2: Data Switching and Graph Update Check ---
Switching datasets while keeping virtual addresses unchanged...
Mapped physical memory to VA: 0x7f5a3d200000
Mapped physical memory to VA: 0x7f5a3d400000
Virtual address verification:
  A: 0x7f5a3d200000 -> 0x7f5a3d200000 (same: YES)
  B: 0x7f5a3d400000 -> 0x7f5a3d400000 (same: YES)

Capturing new graph with updated data...
New graph captured successfully

Checking if existing graph_exec can be updated with new graph...
Update check completed in 5us
Update result: Success (code: 0)

cudaGraphExecUpdate SUCCESS!
Meaning: The existing graph_exec can handle the new data without recompilation.
Reason: Only memory content changed, addresses remain identical.

Reusing existing graph_exec with new data...
Graph reuse execution time: 36us

Verifying results with new data...
Phase 2: Results are correct

Data change verification:
  Phase 1 result sum (first 10): 41880
  Phase 2 result sum (first 10): 365409
  Difference: 323529
Confirmed: Computation used new dataset
Graph successfully reused: YES

Demo completed successfully!

```