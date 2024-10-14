// Inspired by TRT-LLM and vLLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM
//   Serving}, author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and
//   Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "../common/dispatch_utils.h"
#include "../common/reduction_utils.cuh"
#include "../common/utils.cuh"
#include "layernorm.h"

namespace vllm {
// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, typename out_type, bool use_quant>
__global__ void rms_norm_kernel(
    out_type *__restrict__ out,           // [..., hidden_size]
    const scalar_t *__restrict__ input,   // [..., hidden_size]
    const scalar_t *__restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    if constexpr (use_quant) {
      out[blockIdx.x * hidden_size + idx] =
          float_to_int8_rn(((float)(x * s_variance)) * (float)(weight[idx]));
    } else {
      out[blockIdx.x * hidden_size + idx] =
          ((scalar_t)(x * s_variance)) * weight[idx];
    }
  }
}
}  // namespace vllm

void rms_norm(torch::Tensor &out,     // [..., hidden_size]
              torch::Tensor &input,   // [..., hidden_size]
              torch::Tensor &weight,  // [hidden_size]
              float epsilon, bool use_quant) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    if (use_quant) {
      vllm::rms_norm_kernel<scalar_t, int8_t, true><<<grid, block, 0, stream>>>(
          out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
          weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    } else {
      vllm::rms_norm_kernel<scalar_t, scalar_t, false>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    }
  });
}
