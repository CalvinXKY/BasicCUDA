# Intro
This project dedicated to help people learning CUDA program. It is different from 
NVIDIA official docs which might need you spend lots of time to read and understand.
Instead, the project connect with real scenarios and brief explanation to make you 
get knowledge as easy as possible. Some parts of the project have different versions, 
which help readers understand them gradually and know how to optimize a kernel/function at the same time.
To make project compile simple, each part is individual and has its own file. 

eg. 
```
cd matrix_multiply 
make
./matMul
```

# 🔍 中文博客
| 📚 文章                                                                                            | 📖 类型  | 🧩 代码                                             |
|:-------------------------------------------------------------------------------------------------|:-------|:--------------------------------------------------|
| [GPU硬件: Tesla 经典架构详解](https://zhuanlan.zhihu.com/p/508862848)                                    |  GPU基础   | -                                                 |
| [GPU硬件：AI算力GPU发展简史](https://zhuanlan.zhihu.com/p/515584277)                                      |  GPU基础   | [link]()                                          |
| [GPU软件：GPU内存(显存)的理解与基本使用](https://zhuanlan.zhihu.com/p/462191421)                                |  GPU基础   | [link](./memory_opt)                              |
| [GPU硬件: MIG-GPU简介与A100-MIG实践详解](https://zhuanlan.zhihu.com/p/558046644)                          |  GPU基础   | -                                                 |
| [GPU硬件: Tensor core和cuda core是什么区别？](https://www.zhihu.com/question/451127498/answer/1813864500) |  GPU基础   | [link]()                                          |
| [GPU硬件: Ampere架构硬件分析与A100测试](https://zhuanlan.zhihu.com/p/559578692)                             |  GPU基础   | [link]()                                          |
| [CUDA全局坐标计算&Grid/Block/threadIdx映射处理](https://zhuanlan.zhihu.com/p/675603584)                    | CUDA C++| [link](./common_methods/threads_hierarchy_calc.cu)|
| [CUDA入门：矩阵乘运算从CPU到GPU](https://zhuanlan.zhihu.com/p/573271688)                                   |  CUDA C++  | [link](./matrix_multiply)                         |
| [CUDA实践：训练融合运算ScaledMaskSoftmax算子](https://zhuanlan.zhihu.com/p/675794183)                       |  CUDA C++  | [link](./transformer/fused_softmax)               |
| [CUDA入门：常用技巧/方法](https://zhuanlan.zhihu.com/p/584501634)                                         |  CUDA C++  | [link](./common_methods)                          |
| [CUDA实践：20行代码入门PyTorch自定义CUDA/C++](https://zhuanlan.zhihu.com/p/579395211)                       |  CUDA C++  | [link](./pytorch/torch_ext)                       |
| [NCCL算法的拓扑建立与通路选择](https://zhuanlan.zhihu.com/p/735606197)                                       |   GPU网络  | [link](./nccl)                                    |
| [NCCL初始化日志解读](https://zhuanlan.zhihu.com/p/719917835)                                            |  GPU网络  | -                                                 |
| [NCCL通信C++示例（一）: 基础用例解读与运行](https://zhuanlan.zhihu.com/p/718639633)                              |   GPU网络  | [link](./nccl)                                    |
| [NCCL通信C++示例（二）: 用socket建立多机连接](https://zhuanlan.zhihu.com/p/718040976)                          |  GPU网络  | [link](./nccl)                                    |
| [NCCL通信C++示例（三）: 多流并发通信（非阻塞）](https://zhuanlan.zhihu.com/p/716805174)                            |  GPU网络  | [link](./nccl)                                    |
| [NCCL通信C++示例（四）: AlltoAll_Split实现与分析](https://zhuanlan.zhihu.com/p/718765726)                    |  GPU网络  | [link](./nccl)                                    |
| [GPU组网：一图了解GPU网络拓扑](https://zhuanlan.zhihu.com/p/678903640)                                      |  GPU基础   | -                                                 |
| [PyTorch显存管理介绍与源码解析（一）](https://zhuanlan.zhihu.com/p/680769942)                                  |   PyTorch  | [link](./pytorch/torch1.13_mem_rationale)         |
| [PyTorch显存管理介绍与源码解析（二）](https://zhuanlan.zhihu.com/p/681651660)                                  |   PyTorch  | [link](./pytorch/torch1.13_mem_rationale)         |
| [PyTorch显存管理介绍与源码解析（三）](https://zhuanlan.zhihu.com/p/692614846)                                  |   PyTorch  | [link](./pytorch/torch1.13_mem_rationale)         |
| [PyTorch显存可视化与Snapshot数据分析](https://zhuanlan.zhihu.com/p/677203832)                              |   PyTorch  | [link](./pytorch/torch_mem_snapshot)              |
