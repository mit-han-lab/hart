// Inspired by TRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM
//   Serving}, author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and
//   Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#include <torch/extension.h>

void rms_norm(torch::Tensor &out,     // [num_tokens, hidden_size]
              torch::Tensor &input,   // [num_tokens, hidden_size]
              torch::Tensor &weight,  // [hidden_size]
              float epsilon, bool use_quant);
