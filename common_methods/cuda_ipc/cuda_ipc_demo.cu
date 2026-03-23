/*
  CUDA IPC 跨进程共享显存最小示例（C++ / nvcc）

  用法（Linux 两个终端）：
  1) 启动发送端（创建显存并导出 IPC 句柄）
     ./cuda_ipc_cpp_demo --mode sender --prefix /tmp/ipc_demo

  2) 启动接收端（打开 IPC 句柄并原地修改）
     ./cuda_ipc_cpp_demo --mode receiver --prefix /tmp/ipc_demo

  预期：
  - sender 初始数据为 0,1,2,3,...
  - receiver 对共享数据执行 +100
  - sender 等待 receiver 完成后读回数据，变为 100,101,102,103,...

  编译（Linux, Driver 570 + CUDA 13.1）:
    nvcc -std=c++14 -O2 cuda_ipc_cpp_demo.cu -o cuda_ipc_cpp_demo

   Author: kaiyuan
   Email: kaiyuanxie@yeah.net
*/

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#define CHECK_CUDA(expr)                                                         \
  do {                                                                           \
    cudaError_t _err = (expr);                                                   \
    if (_err != cudaSuccess) {                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                     \
                               cudaGetErrorString(_err) + " @ " + #expr);        \
    }                                                                            \
  } while (0)

struct IPCPacket {
  cudaIpcMemHandle_t mem_handle;
  int device = 0;
  int n = 0;
};

__global__ void init_kernel(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = static_cast<float>(i);
}

__global__ void add_kernel(float* p, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += v;
}

static bool file_exists(const std::string& path) {
  std::ifstream fin(path, std::ios::binary);
  return fin.good();
}

static void write_packet(const std::string& path, const IPCPacket& pkt) {
  std::ofstream fout(path, std::ios::binary | std::ios::trunc);
  if (!fout) throw std::runtime_error("failed to open file for write: " + path);
  fout.write(reinterpret_cast<const char*>(&pkt), sizeof(pkt));
  if (!fout) throw std::runtime_error("failed to write packet to: " + path);
}

static IPCPacket read_packet(const std::string& path) {
  IPCPacket pkt{};
  std::ifstream fin(path, std::ios::binary);
  if (!fin) throw std::runtime_error("failed to open file for read: " + path);
  fin.read(reinterpret_cast<char*>(&pkt), sizeof(pkt));
  if (!fin) throw std::runtime_error("failed to read packet from: " + path);
  return pkt;
}

static void touch_file(const std::string& path) {
  std::ofstream fout(path, std::ios::binary | std::ios::trunc);
  if (!fout) throw std::runtime_error("failed to create file: " + path);
}

static void remove_file_if_exists(const std::string& path) {
  if (file_exists(path)) {
    std::remove(path.c_str());
  }
}

static void wait_file(const std::string& path, int sleep_ms = 100) {
  while (!file_exists(path)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  }
}

static void sender(const std::string& prefix) {
  constexpr int n = 16;
  const std::string handle_path = prefix + ".handle.bin";
  const std::string done_path = prefix + ".done.flag";
  remove_file_if_exists(handle_path);
  remove_file_if_exists(done_path);

  int device = 0;
  CHECK_CUDA(cudaSetDevice(device));

  float* d_ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ptr, n * sizeof(float)));

  int threads = 128;
  int blocks = (n + threads - 1) / threads;
  init_kernel<<<blocks, threads>>>(d_ptr, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaIpcMemHandle_t mem_handle{};
  CHECK_CUDA(cudaIpcGetMemHandle(&mem_handle, d_ptr));

  IPCPacket pkt{};
  pkt.mem_handle = mem_handle;
  pkt.device = device;
  pkt.n = n;
  write_packet(handle_path, pkt);
  std::cout << "[sender] wrote IPC handle to: " << handle_path << std::endl;
  std::cout << "[sender] waiting receiver done flag: " << done_path << std::endl;

  wait_file(done_path);

  float h[n];
  CHECK_CUDA(cudaMemcpy(h, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::cout << "[sender] values after receiver modify: ";
  for (int i = 0; i < 8; ++i) std::cout << h[i] << (i + 1 == 8 ? "" : ", ");
  std::cout << std::endl;

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expect = static_cast<float>(i + 100);
    if (h[i] != expect) {
      ok = false;
      break;
    }
  }
  std::cout << (ok ? "[sender] PASS: shared GPU memory verified"
                   : "[sender] FAIL: values mismatch")
            << std::endl;

  CHECK_CUDA(cudaFree(d_ptr));
}

static void receiver(const std::string& prefix) {
  const std::string handle_path = prefix + ".handle.bin";
  const std::string done_path = prefix + ".done.flag";
  remove_file_if_exists(done_path);

  std::cout << "[receiver] waiting handle file: " << handle_path << std::endl;
  wait_file(handle_path);
  IPCPacket pkt = read_packet(handle_path);

  CHECK_CUDA(cudaSetDevice(pkt.device));

  void* base_ptr = nullptr;
  CHECK_CUDA(cudaIpcOpenMemHandle(&base_ptr, pkt.mem_handle,
                                  cudaIpcMemLazyEnablePeerAccess));

  float* d_ptr = static_cast<float*>(base_ptr);
  int threads = 128;
  int blocks = (pkt.n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(d_ptr, pkt.n, 100.0f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaIpcCloseMemHandle(base_ptr));
  touch_file(done_path);
  std::cout << "[receiver] modify done, wrote flag: " << done_path << std::endl;
}

static std::string get_arg(int argc, char** argv, const std::string& key,
                           const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) return argv[i + 1];
  }
  return def;
}

int main(int argc, char** argv) {
  try {
    std::string mode = get_arg(argc, argv, "--mode", "");
    std::string prefix = get_arg(argc, argv, "--prefix", "cuda_ipc_cpp_demo");

    if (mode != "sender" && mode != "receiver") {
      std::cerr << "Usage: " << argv[0]
                << " --mode sender|receiver [--prefix ipc_demo]\n";
      return 1;
    }

    if (mode == "sender")
      sender(prefix);
    else
      receiver(prefix);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << std::endl;
    return 2;
  }
}
