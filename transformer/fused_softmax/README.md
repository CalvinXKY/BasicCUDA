
# Fused softmax

## Formula： 

The fuesd opts including：
* opt1: softmax
* opt2: scale
* opt3: mask

**1 Softmax**
 
* forward:  yi=e^{xi- max(X)}/\sum_{j=1}^{n}{e^{xj- max(X)}}
* backward: dxi = yi * d yi - yi * \sum_{j=1}^{n}{yj * dyj}  

**2 Scale**

output = input * scale

**3 Mask**
```textmate
if(mask[i] == 1) 
then 
   val[i] = -VAL 
else 
   do_something
```

input data shape: 

[batches, attn_heads, query_seq_len, key_seq_len]



## Requirements

pytorch>=2.0

cuda>=11.3

hardware: GPU >= volta

## compile

```
python setup.py build
```

## Running

### Function invoke: 
```python
import transformer_softmax_lib
# ...
transformer_softmax_lib.scaled_masked_softmax_forward(input_data, mask, scale_factor)
```

### A test example:

note:make sure the .so file is in your running dircetion:

```python
import torch
import transformer_softmax_lib
from torch.autograd import Function

class FusedSoftmax(Function):
    @staticmethod
    def forward(ctx, src, mask, scale_factor):

        output = transformer_softmax_lib.scaled_masked_softmax_forward(src, mask, scale_factor[0])
        ctx.save_for_backward(output , scale_factor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        src, scale_factor = ctx.saved_tensors
        grad_in = transformer_softmax_lib.scaled_masked_softmax_backward(grad_output, src, scale_factor[0])
        return grad_in, None, None  # 与输入对应上。

data_input = torch.randn([1,8,1024,1024], dtype=torch.float16, device='cuda', requires_grad=True)
data_input_check = data_input.clone().detach()
data_input_check.requires_grad_(True)
factor = torch.tensor([1.0], requires_grad=False)
mask = torch.zeros([1,1,1024,1024], dtype=torch.float16, device='cuda', requires_grad=False)
check = torch.softmax(data_input_check, dim=-1)
out_put = FusedSoftmax.apply(data_input, mask, factor)

# forward check:
print(torch.allclose(check, out_put, atol=1e-05, rtol=1e-05 )) # fp16 

# backward check：
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
    y=out_put.sum().backward()
    y_check=check.sum().backward()
print(torch.allclose(data_input.grad, data_input_check.grad, atol=1e-05, rtol=1e-05 ))
```
