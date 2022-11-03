# torch cuda custom example #

import math
from torch import nn
from torch.autograd import Function
import torch
import time
try:
    import lltm_cuda
except ImportError as e:
    print("lltm_cuda.so is not found! Use JIT compiling....")
    from torch.utils.cpp_extension import load
    lltm_cuda = load(
        'lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu']) # verbose=True 
    print("lltm_cuda dir:", lltm_cuda.__file__)


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cuda.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cuda.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float32
    kwargs = {'dtype': dtype,
              'device': device,
              'requires_grad': True}
    batch_size = 32
    features = 32
    state_size = 256
    iter_nums = 100

    X = torch.randn(batch_size, features, **kwargs)
    h = torch.randn(batch_size, state_size, **kwargs)
    C = torch.randn(batch_size, state_size, **kwargs)
    rnn = LLTM(features, state_size).to(device, dtype)
    # Force CUDA initialization
    new_h, new_C = rnn(X, (h, C))
    (new_h.sum() + new_C.sum()).backward()

    forward_min = math.inf
    forward_time = 0
    backward_min = math.inf
    backward_time = 0

    for _ in range(iter_nums):
        rnn.zero_grad()
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        elapsed = time.time() - start
        backward_min = min(backward_min, elapsed)
        backward_time += elapsed

    forward_min *= 1000
    backward_min *= 1000
    forward_average = forward_time / iter_nums * 1000
    backward_average = backward_time / iter_nums * 1000

    print("Custom lltm_cuda result: ")
    print('Forward: min:{0:.3f} ms avg:{1:.3f} ms | Backward min: {2:.3f} '
          'ms avg: {3:.3f} ms'.format(forward_min, forward_average,
                                      backward_min, backward_average,))
