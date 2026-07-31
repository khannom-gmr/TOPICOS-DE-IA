#pragma once
#include "cuda_tensor.cuh"
#include "cuda_layers.cuh"
#include <cublas_v2.h>

namespace vit::cuda {

class CudaMultiHeadAttention {
public:
    int embed_dim_, num_heads_, head_dim_;

    CudaLinear W_q, W_k, W_v, W_o;

    CudaMultiHeadAttention() = default;
    CudaMultiHeadAttention(int embed_dim, int num_heads);

    CudaTensor forward(cublasHandle_t handle, const CudaTensor& x,
                       int B, int N);

    CudaTensor backward(cublasHandle_t handle, const CudaTensor& grad_out,
                        int B, int N);

    void zero_grad();

    void get_params(std::vector<CudaTensor*>& params,
                    std::vector<CudaTensor*>& grads);

private:
    CudaTensor last_x_;
    CudaTensor last_Q_;
    CudaTensor last_K_;
    CudaTensor last_V_;
    CudaTensor last_attn_;
    CudaTensor last_ctx_heads_;

    CudaTensor split_heads_device(const CudaTensor& x_flat, int B, int N) const;
    CudaTensor merge_heads_device(const CudaTensor& heads, int B, int N) const;
};

} // namespace vit::cuda
