import torch
from torch.utils.cpp_extension import load

ext_module = load(name="demo",  sources=["demo.cu"])
ext_module.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
