# torch cuda custom example #

import math
import time

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F


def d_sigmoid(z):
    s = torch.sigmoid(z)
    return (1 - s) * s


def d_tanh(z):
    t = torch.tanh(z)
    return 1 - (t * t)


def d_elu(z, alpha=1.0):
    e = z.exp()
    mask = (alpha * (e - 1)) < 0
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e)


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        X = torch.cat([old_h, input], dim=1)

        gate_weights = F.linear(X, weights, bias)
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        candidate_cell = F.elu(gates[2])

        new_cell = old_cell + candidate_cell * input_gate
        new_h = torch.tanh(new_cell) * output_gate

        ctx.save_for_backward(X, weights, input_gate, output_gate, old_cell,
                              new_cell, candidate_cell, gate_weights)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        X, weights, input_gate, output_gate, old_cell = ctx.saved_variables[:5]
        new_cell, candidate_cell, gate_weights = ctx.saved_variables[5:]

        d_input = d_weights = d_bias = d_old_h = d_old_cell = None

        d_output_gate = torch.tanh(new_cell) * grad_h
        d_tanh_new_cell = output_gate * grad_h
        d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell

        d_old_cell = d_new_cell
        d_candidate_cell = input_gate * d_new_cell
        d_input_gate = candidate_cell * d_new_cell

        gates = gate_weights.chunk(3, dim=1)
        d_input_gate *= d_sigmoid(gates[0])
        d_output_gate *= d_sigmoid(gates[1])
        d_candidate_cell *= d_elu(gates[2])

        d_gates = torch.cat(
            [d_input_gate, d_output_gate, d_candidate_cell], dim=1)

        if ctx.needs_input_grad[1]:
            d_weights = d_gates.t().mm(X)
        if ctx.needs_input_grad[2]:
            d_bias = d_gates.sum(dim=0, keepdim=True)
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            d_X = d_gates.mm(weights)
            state_size = grad_h.shape[1]
            d_old_h, d_input = d_X[:, :state_size], d_X[:, state_size:]

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

    print("PyTorch baseline result:")
    print('Forward: min:{0:.3f} ms avg:{1:.3f} ms | Backward min: {2:.3f} '
          'ms avg: {3:.3f} ms'.format(forward_min, forward_average,
                                      backward_min, backward_average,))
