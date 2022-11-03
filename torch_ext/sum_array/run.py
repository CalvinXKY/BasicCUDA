import torch
import time
from torch.utils.cpp_extension import load

ext_module = load(name="sum_array",
                  extra_include_paths=["./"] ,
                  sources=["sumArray.cu", "glueCode.cpp"],
                  verbose=True)


def iter_test(func):
    delta_t = 0
    for _ in range(10000):
        _tensor = torch.rand(50000, dtype=torch.float, device='cuda')
        t1 = time.time()
        func(_tensor)
        t2 = time.time()
        delta_t += t2-t1
    print("    Elapsed time:", delta_t)
    return delta_t


if __name__ == "__main__":
    # warm up:
    in_tensor = torch.rand(50000, dtype=torch.float, device='cuda')
    print(torch.sum(in_tensor))
    print(ext_module.sum_array(in_tensor.clone()))

    # time test:
    print("The torch original sum func test:")
    iter_test(torch.sum)
    print("The custom define sum func test:")
    iter_test(ext_module.sum_array)
    