#include "vit/attention.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace vit {

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim_(embed_dim), num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      W_q(embed_dim, embed_dim),
      W_k(embed_dim, embed_dim),
      W_v(embed_dim, embed_dim),
      W_o(embed_dim, embed_dim) {}

Tensor MultiHeadAttention::split_heads(const Tensor& x) const {
    int B = x.shape[0], N = x.shape[1];
    Tensor out({B, num_heads_, N, head_dim_});
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < num_heads_; ++h)
                for (int d = 0; d < head_dim_; ++d)
                    out.at(b, h, n, d) = x.at(b, n, h * head_dim_ + d);
    return out;
}

Tensor MultiHeadAttention::merge_heads(const Tensor& x) const {
    int B = x.shape[0], N = x.shape[2];
    Tensor out({B, N, embed_dim_});
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < num_heads_; ++h)
                for (int d = 0; d < head_dim_; ++d)
                    out.at(b, n, h * head_dim_ + d) = x.at(b, h, n, d);
    return out;
}

static Tensor attention_softmax(const Tensor& scores) {
    int B = scores.shape[0], H = scores.shape[1];
    int N = scores.shape[2];
    int M = scores.shape[3];
    Tensor out(scores.shape);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < N; ++i) {
                float max_v = -1e38f;
                for (int j = 0; j < M; ++j)
                    max_v = std::max(max_v, scores.at(b, h, i, j));
                float sum = 0.f;
                for (int j = 0; j < M; ++j) {
                    float e = std::exp(scores.at(b, h, i, j) - max_v);
                    out.at(b, h, i, j) = e;
                    sum += e;
                }
                float inv = 1.f / sum;
                for (int j = 0; j < M; ++j)
                    out.at(b, h, i, j) *= inv;
            }
    return out;
}

static Tensor attention_softmax_backward(const Tensor& A, const Tensor& dA) {
    int B = A.shape[0], H = A.shape[1], N = A.shape[2], M = A.shape[3];
    Tensor dS(A.shape);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < N; ++i) {
                float dot = 0.f;
                for (int j = 0; j < M; ++j)
                    dot += A.at(b, h, i, j) * dA.at(b, h, i, j);
                for (int j = 0; j < M; ++j)
                    dS.at(b, h, i, j) = A.at(b, h, i, j) * (dA.at(b, h, i, j) - dot);
            }
    return dS;
}

static Tensor bmm4(const Tensor& A, const Tensor& B) {
    int batch = A.shape[0], H = A.shape[1], M = A.shape[2], K = A.shape[3];
    int N = B.shape[3];
    Tensor C({batch, H, M, N});
    for (int b = 0; b < batch; ++b)
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < M; ++i)
                for (int k = 0; k < K; ++k) {
                    float a = A.at(b, h, i, k);
                    if (a == 0.f) continue;
                    for (int j = 0; j < N; ++j)
                        C.at(b, h, i, j) += a * B.at(b, h, k, j);
                }
    return C;
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    last_x_ = x;
    int B = x.shape[0], N = x.shape[1];

    Tensor x_flat = x.reshape({B * N, embed_dim_});

    Tensor Q_flat = W_q.forward(x_flat);
    Tensor K_flat = W_k.forward(x_flat);
    Tensor V_flat = W_v.forward(x_flat);

    last_Q_ = split_heads(Q_flat.reshape({B, N, embed_dim_}));
    last_K_ = split_heads(K_flat.reshape({B, N, embed_dim_}));
    last_V_ = split_heads(V_flat.reshape({B, N, embed_dim_}));

    Tensor K_T = last_K_.transpose_4d_23();
    float scale = 1.f / std::sqrt(static_cast<float>(head_dim_));

    Tensor scores = bmm4(last_Q_, K_T);
    for (auto& v : scores.data) v *= scale;
    last_attn_in_ = scores;

    last_attn_ = attention_softmax(scores);

    Tensor ctx_heads = bmm4(last_attn_, last_V_);

    Tensor ctx = merge_heads(ctx_heads).reshape({B * N, embed_dim_});
    last_ctx_ = ctx;

    Tensor out_flat = W_o.forward(ctx);
    return out_flat.reshape({B, N, embed_dim_});
}

Tensor MultiHeadAttention::backward(const Tensor& grad_out) {
    int B = last_x_.shape[0], N = last_x_.shape[1];

    Tensor grad_out_flat = grad_out.reshape({B * N, embed_dim_});
    Tensor d_ctx = W_o.backward(grad_out_flat);

    Tensor d_ctx_heads = split_heads(d_ctx.reshape({B, N, embed_dim_}));

    Tensor V_T    = last_V_.transpose_4d_23();
    Tensor d_attn = bmm4(d_ctx_heads, V_T);

    Tensor attn_T = last_attn_.transpose_4d_23();
    Tensor d_V    = bmm4(attn_T, d_ctx_heads);

    Tensor d_scores = attention_softmax_backward(last_attn_, d_attn);

    float scale = 1.f / std::sqrt(static_cast<float>(head_dim_));
    for (auto& v : d_scores.data) v *= scale;

    Tensor d_Q = bmm4(d_scores, last_K_);
    Tensor d_scores_T = d_scores.transpose_4d_23();
    Tensor d_K = bmm4(d_scores_T, last_Q_);

    Tensor d_Q_flat = merge_heads(d_Q).reshape({B * N, embed_dim_});
    Tensor d_K_flat = merge_heads(d_K).reshape({B * N, embed_dim_});
    Tensor d_V_flat = merge_heads(d_V).reshape({B * N, embed_dim_});

    Tensor d_x_q = W_q.backward(d_Q_flat);
    Tensor d_x_k = W_k.backward(d_K_flat);
    Tensor d_x_v = W_v.backward(d_V_flat);

    Tensor d_x_flat(d_x_q.shape);
    for (int i = 0; i < d_x_flat.numel(); ++i)
        d_x_flat.data[i] = d_x_q.data[i] + d_x_k.data[i] + d_x_v.data[i];

    return d_x_flat.reshape({B, N, embed_dim_});
}

void MultiHeadAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    return {&W_q.W, &W_q.b, &W_k.W, &W_k.b,
            &W_v.W, &W_v.b, &W_o.W, &W_o.b};
}

std::vector<Tensor*> MultiHeadAttention::gradients() {
    return {&W_q.dW, &W_q.db, &W_k.dW, &W_k.db,
            &W_v.dW, &W_v.db, &W_o.dW, &W_o.db};
}

} // namespace vit
