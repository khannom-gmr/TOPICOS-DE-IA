#pragma once
#include "vit/tensor.hpp"
#include "vit/layers.hpp"

namespace vit {

// -----------------------------------------------------------------------
// Multi-Head Self-Attention (MHSA) - Slides 40-77
//
//   1. Proyeccion QKV: Q = X W_q^T, K = X W_k^T, V = X W_v^T
//   2. Similaridad QK: E = Q K^T / sqrt(D_H)
//   3. Ponderacion V: A = softmax(E), Y = A V
//   4. Proyeccion Salida: O = Y W_o^T
// -----------------------------------------------------------------------
class MultiHeadAttention {
public:
    int embed_dim_;
    int num_heads_;
    int head_dim_;

    Linear W_q, W_k, W_v, W_o;

    MultiHeadAttention() = default;
    MultiHeadAttention(int embed_dim, int num_heads);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);

    void zero_grad();

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

private:
    Tensor last_x_;
    Tensor last_Q_;
    Tensor last_K_;
    Tensor last_V_;
    Tensor last_attn_;
    Tensor last_attn_in_;
    Tensor last_ctx_;

    Tensor split_heads(const Tensor& x) const;  // [B,N,D] -> [B,H,N,d]
    Tensor merge_heads(const Tensor& x) const;  // [B,H,N,d] -> [B,N,D]
};

} // namespace vit
