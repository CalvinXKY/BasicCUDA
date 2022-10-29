/**
 *  PyTorch extension cuda example: sum array.
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */

#pragma once

// CUDA runtime
#include <cuda_runtime.h>
#define THREAD_PER_BLOCK 256


void arraySumCUDA(float *, const int);