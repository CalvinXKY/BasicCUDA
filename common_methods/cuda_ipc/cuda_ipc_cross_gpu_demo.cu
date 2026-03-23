/*
  CUDA IPC 跨卡读写示例（同一节点两个独立进程）

  目标：
  - sender 在 src-device 上分配显存并导出 IPC handle
  - receiver 在 access-device 上打开该 handle
  - receiver 在 access-device 上发 kernel，直接修改 src-device 的那块显存
  - sender 读回验证数据变化，证明跨卡远程读写成功

  Linux 编译：
    nvcc -std=c++14 -O2 cuda_ipc_cross_gpu_demo.cu -o cuda_ipc_cross_gpu_demo

  Linux 运行（两个终端）：
    # 终端1（先启动接收端）
    ./cuda_ipc_cross_gpu_demo --mode receiver --prefix /tmp/ipc_xgpu --src-device 0 --access-device 1

    # 终端2（再启动发送端）
    ./cuda_ipc_cross_gpu_demo --mode sender --prefix /tmp/ipc_xgpu --src-device 0 --access-device 1

   Author: kaiyuan
   Email: kaiyuanxie@yeah.net
*/

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = (expr);                                                  \
    if (_err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error: ") +                    \
                               cudaGetErrorString(_err) + " @ " + #expr);       \
    }                                                                           \
  } while (0)

struct IPCPacket {
  cudaIpcMemHandle_t mem_handle;
  int src_device = 0;
  int access_device = 0;
  int n = 0;
};

__global__ void init_kernel(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = static_cast<float>(i);
  }
}

__global__ void add_kernel(float* p, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] += v;
  }
}

static bool file_exists(const std::string& path) {
  std::ifstream fin(path, std::ios::binary);
  return fin.good();
}

static void remove_file_if_exists(const std::string& path) {
  if (file_exists(path)) {
    std::remove(path.c_str());
  }
}

static void write_packet(const std::string& path, const IPCPacket& pkt) {
  std::ofstream fout(path, std::ios::binary | std::ios::trunc);
  if (!fout) {
    throw std::runtime_error("failed to open for write: " + path);
  }
  fout.write(reinterpret_cast<const char*>(&pkt), sizeof(pkt));
  if (!fout) {
    throw std::runtime_error("failed to write packet: " + path);
  }
}

static IPCPacket read_packet(const std::string& path) {
  IPCPacket pkt{};
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("failed to open for read: " + path);
  }
  fin.read(reinterpret_cast<char*>(&pkt), sizeof(pkt));
  if (!fin) {
    throw std::runtime_error("failed to read packet: " + path);
  }
  return pkt;
}

static void touch_file(const std::string& path) {
  std::ofstream fout(path, std::ios::binary | std::ios::trunc);
  if (!fout) {
    throw std::runtime_error("failed to create file: " + path);
  }
}

static void wait_file(const std::string& path, int sleep_ms = 100) {
  while (!file_exists(path)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  }
}

static int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      return std::stoi(argv[i + 1]);
    }
  }
  return def;
}

static std::string parse_str_arg(int argc, char** argv, const std::string& key,
                                 const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      return argv[i + 1];
    }
  }
  return def;
}

static void check_devices_and_peer(int src_device, int access_device) {
  int dev_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&dev_count));
  if (src_device < 0 || src_device >= dev_count) {
    throw std::runtime_error("src-device out of range");
  }
  if (access_device < 0 || access_device >= dev_count) {
    throw std::runtime_error("access-device out of range");
  }

  int can_access = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, access_device, src_device));
  if (!can_access) {
    throw std::runtime_error(
        "access-device cannot peer-access src-device; cross-GPU IPC read/write not supported on this topology");
  }
}

static void run_sender(const std::string& prefix, int src_device, int access_device) {
  constexpr int n = 16;
  const std::string handle_path = prefix + ".handle.bin";
  const std::string done_path = prefix + ".done.flag";

  check_devices_and_peer(src_device, access_device);
  remove_file_if_exists(handle_path);
  remove_file_if_exists(done_path);

  CHECK_CUDA(cudaSetDevice(src_device));

  float* d_ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ptr, n * sizeof(float)));

  int threads = 128;
  int blocks = (n + threads - 1) / threads;
  init_kernel<<<blocks, threads>>>(d_ptr, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaIpcMemHandle_t handle{};
  CHECK_CUDA(cudaIpcGetMemHandle(&handle, d_ptr));

  IPCPacket pkt{};
  pkt.mem_handle = handle;
  pkt.src_device = src_device;
  pkt.access_device = access_device;
  pkt.n = n;
  write_packet(handle_path, pkt);
  std::cout << "[sender] wrote handle: " << handle_path << std::endl;
  std::cout << "[sender] src-device=" << src_device
            << ", waiting remote write from access-device=" << access_device
            << std::endl;

  wait_file(done_path);

  float h[n];
  CHECK_CUDA(cudaMemcpy(h, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::cout << "[sender] values after receiver write: ";
  for (int i = 0; i < 8; ++i) {
    std::cout << h[i] << (i + 1 == 8 ? "" : ", ");
  }
  std::cout << std::endl;

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    const float expect = static_cast<float>(i + 1000);
    if (h[i] != expect) {
      ok = false;
      break;
    }
  }
  std::cout << (ok ? "[sender] PASS: cross-GPU IPC remote write verified"
                   : "[sender] FAIL: value check mismatch")
            << std::endl;

  CHECK_CUDA(cudaFree(d_ptr));
}

static void run_receiver(const std::string& prefix, int src_device, int access_device) {
  const std::string handle_path = prefix + ".handle.bin";
  const std::string done_path = prefix + ".done.flag";

  check_devices_and_peer(src_device, access_device);
  remove_file_if_exists(done_path);

  std::cout << "[receiver] waiting handle: " << handle_path << std::endl;
  wait_file(handle_path);
  IPCPacket pkt = read_packet(handle_path);

  if (pkt.src_device != src_device || pkt.access_device != access_device) {
    throw std::runtime_error("packet device config mismatch with cli args");
  }

  CHECK_CUDA(cudaSetDevice(access_device));

  void* remote_ptr = nullptr;
  CHECK_CUDA(cudaIpcOpenMemHandle(&remote_ptr, pkt.mem_handle,
                                  cudaIpcMemLazyEnablePeerAccess));

  float* d_remote = static_cast<float*>(remote_ptr);
  int threads = 128;
  int blocks = (pkt.n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(d_remote, pkt.n, 1000.0f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaIpcCloseMemHandle(remote_ptr));
  touch_file(done_path);
  std::cout << "[receiver] remote write done on access-device=" << access_device
            << ", modified src-device=" << src_device << " memory" << std::endl;
}

int main(int argc, char** argv) {
  try {
    std::string mode = parse_str_arg(argc, argv, "--mode", "");
    std::string prefix = parse_str_arg(argc, argv, "--prefix", "/tmp/ipc_xgpu");
    int src_device = parse_int_arg(argc, argv, "--src-device", 0);
    int access_device = parse_int_arg(argc, argv, "--access-device", 1);

    if (mode != "sender" && mode != "receiver") {
      std::cerr << "Usage: " << argv[0]
                << " --mode sender|receiver --prefix /tmp/ipc_xgpu"
                << " --src-device 0 --access-device 1\n";
      return 1;
    }

    if (mode == "sender") {
      run_sender(prefix, src_device, access_device);
    } else {
      run_receiver(prefix, src_device, access_device);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << std::endl;
    return 2;
  }
}
