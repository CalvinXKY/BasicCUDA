
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

```python
import transformer_softmax_lib
# ...
transformer_softmax_lib.scaled_masked_softmax_forward(input_data, mask, scale_factor)
```
