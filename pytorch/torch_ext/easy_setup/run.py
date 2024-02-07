import torch
import my_extension


my_extension.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
