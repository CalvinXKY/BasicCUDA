# CUDA IPC C++ 示例说明

这个示例演示了**两个独立进程**如何通过CUDA IPC（Inter-Process Communication）共享同一块GPU显存，并完成跨进程原地修改。

## 1. CUDA IPC 原理（简版）

CUDA IPC的核心不是“拷贝数据到另一个进程”，而是：

1. 发送进程先在GPU上`cudaMalloc`一块显存；
2. 发送进程把这块显存导出成 `cudaIpcMemHandle_t`（句柄）；
3. 接收进程拿到句柄后，在自己的进程上下文里“打开”这块显存；
4. 两个进程都能访问同一块物理显存（只是各自进程里的虚拟地址可能不同）。

因此，接收进程对这块内存做原地写入，发送进程随后读回时会看到变化。

## 2. 关键接口与用途

- `cudaIpcGetMemHandle(&handle, d_ptr)`  
  发送端：把`cudaMalloc`出来的设备指针导出为IPC句柄。

- `cudaIpcOpenMemHandle(&base_ptr, handle, cudaIpcMemLazyEnablePeerAccess)`  
  接收端：根据句柄打开共享显存，得到本进程可用的设备指针。

- `cudaIpcCloseMemHandle(base_ptr)`  
  接收端：使用结束后关闭映射句柄。

- `cudaDeviceSynchronize()`  
  两端都使用，确保 kernel 执行完成，避免异步导致的“看起来没改成功”。

## 3. 用例简述

### sender（发送端）

1. 在GPU上分配`float`数组；
2. 用 `init_kernel` 初始化为 `0,1,2,...`；
3. 导出IPC句柄并写入文件；
4. 等待receiver完成标记；
5. 把数据从GPU拷回host，检查是否变成`100,101,102,...`。

### receiver（接收端）

1. 等待sender写好句柄文件；
2. 读取句柄并 `cudaIpcOpenMemHandle` 打开共享显存；
3. 运行 `add_kernel(..., +100)` 原地修改这块共享显存；
4. 关闭句柄并写完成标记。

## 4. 运行方式（Linux）

### 编译

```bash
nvcc -std=c++14 -O2 cuda_ipc_demo.cu -o cuda_ipc_demo
```

### 终端 1：接收端

```bash
./cuda_ipc_demo --mode receiver --prefix /tmp/ipc_demo
```

### 终端 2：发送端

```bash
./cuda_ipc_demo --mode sender --prefix /tmp/ipc_demo
```

## 5. 如何判断“确实共享了同一块显存”

看发送端最终输出：

- 若输出类似`values after receiver modify: 100, 101, ...`，并显示`PASS`，说明接收端对共享显存的原地修改被发送端直接观察到了；
- 这就证明了是同一块GPU内存被跨进程共享，而不是两边各自有一份副本。

## 6. 使用注意事项

- sender进程必须在receiver使用期间保持存活（显存所有权在sender）；
- 两个进程通常需要在同一台机器、同一GPU环境下运行；
- 本示例用文件做最小进程间同步，生产环境建议用更稳健的IPC控制通道（Unix Socket、共享内存信号量等）。

## 7. 测试

验证环境：

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

使用nvidia官方镜像运行：[Link](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=pytorch)

输出打印：

发送端：
```
[sender] wrote IPC handle to: /tmp/ipc_demo.handle.bin
[sender] waiting receiver done flag: /tmp/ipc_demo.done.flag
[sender] values after receiver modify: 100, 101, 102, 103, 104, 105, 106, 107
[sender] PASS: shared GPU memory verified
```

接收端：
```
[receiver] waiting handle file: /tmp/ipc_demo.handle.bin
[receiver] modify done, wrote flag: /tmp/ipc_demo.done.flag
```


# 跨卡IPC远程读写示例

对应代码：`cuda_ipc_cross_gpu_demo.cu`。

过程：

- sender在`src-device`（例如GPU0）分配显存并导出IPC句柄；
- receiver在`access-device`（例如GPU1）打开该句柄；
- receiver在GPU1上发kernel，直接修改GPU0上的那块共享显存（远程写）；
- sender在GPU0读回结果并校验变化。

### 跨卡模式的关键点

- 能否跨卡远程访问，取决于设备拓扑与P2P能力；
- 示例里会先调用`cudaDeviceCanAccessPeer(access_device, src_device)`做检查；
- 若不支持，会明确报错并退出。

### 编译（Linux）

```bash
nvcc -std=c++14 -O2 cuda_ipc_cross_gpu_demo.cu -o cuda_ipc_cross_gpu_demo
```

### 运行（两个终端）

终端1（先启动receiver）：

```bash
./cuda_ipc_cross_gpu_demo --mode receiver --prefix /tmp/ipc_xgpu --src-device 0 --access-device 1
```

终端2（再启动sender）：

```bash
./cuda_ipc_cross_gpu_demo --mode sender --prefix /tmp/ipc_xgpu --src-device 0 --access-device 1
```

### 输出

- receiver日志会显示已在`access-device`完成远程写；
- sender日志会显示读回值由`i`变为`i + 1000`；
- 最终输出`PASS: cross-GPU IPC remote write verified`，即可证明跨卡IPC远程读写成功。


发送端：
```
[sender] wrote handle: /tmp/ipc_xgpu.handle.bin
[sender] src-device=0, waiting remote write from access-device=1
[sender] values after receiver write: 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007
[sender] PASS: cross-GPU IPC remote write verified
```

接收端：
'''
[receiver] waiting handle: /tmp/ipc_xgpu.handle.bin
[receiver] remote write done on access-device=1, modified src-device=0 memory
'''