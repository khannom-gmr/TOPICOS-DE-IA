#include "vit/layers.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace vit {

Linear::Linear(int in_features, int out_features) {
    W  = Tensor({out_features, in_features});
    b  = Tensor({out_features}, 0.f);
    dW = Tensor({out_features, in_features}, 0.f);
    db = Tensor({out_features}, 0.f);
    W.kaiming_uniform_(in_features);
}

Tensor Linear::forward(const Tensor& x) {
    last_input_ = x;
    int B   = x.shape[0];
    int in  = x.shape[1];
    int out = W.shape[0];

    Tensor result({B, out});
    for (int bi = 0; bi < B; ++bi)
        for (int o = 0; o < out; ++o) {
            float acc = b.data[o];
            for (int i = 0; i < in; ++i)
                acc += x.at(bi, i) * W.at(o, i);
            result.at(bi, o) = acc;
        }
    return result;
}

Tensor Linear::backward(const Tensor& grad_out) {
    int B   = last_input_.shape[0];
    int in  = last_input_.shape[1];
    int out = W.shape[0];

    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i)
            for (int b = 0; b < B; ++b)
                dW.at(o, i) += grad_out.at(b, o) * last_input_.at(b, i);

    for (int o = 0; o < out; ++o)
        for (int b = 0; b < B; ++b)
            db.at(o) += grad_out.at(b, o);

    Tensor grad_in({B, in});
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < in; ++i)
            for (int o = 0; o < out; ++o)
                grad_in.at(b, i) += grad_out.at(b, o) * W.at(o, i);

    return grad_in;
}

void Linear::zero_grad() { dW.zero_(); db.zero_(); }

LayerNorm::LayerNorm(int dim, float eps) : eps(eps), dim_(dim) {
    gamma = Tensor({dim}, 1.f);
    beta  = Tensor({dim}, 0.f);
    dgamma = Tensor({dim}, 0.f);
    dbeta  = Tensor({dim}, 0.f);
}

Tensor LayerNorm::forward(const Tensor& x) {
    assert(x.ndim() == 3 && x.shape[2] == dim_);
    last_x_    = x;
    int B = x.shape[0], N = x.shape[1], D = dim_;

    last_mean_ = Tensor({B, N}, 0.f);
    last_var_  = Tensor({B, N}, 0.f);
    last_xhat_ = Tensor({B, N, D});
    Tensor out({B, N, D});

    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            float mu = 0.f;
            for (int d = 0; d < D; ++d) mu += x.at(b, n, d);
            mu /= D;
            last_mean_.at(b, n) = mu;

            float var = 0.f;
            for (int d = 0; d < D; ++d) {
                float diff = x.at(b, n, d) - mu;
                var += diff * diff;
            }
            var /= D;
            last_var_.at(b, n) = var;

            float inv_std = 1.f / std::sqrt(var + eps);
            for (int d = 0; d < D; ++d) {
                float xh = (x.at(b, n, d) - mu) * inv_std;
                last_xhat_.at(b, n, d) = xh;
                out.at(b, n, d) = gamma.at(d) * xh + beta.at(d);
            }
        }
    }
    return out;
}

Tensor LayerNorm::backward(const Tensor& grad_out) {
    int B = last_x_.shape[0], N = last_x_.shape[1], D = dim_;
    Tensor grad_in({B, N, D});

    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            float var     = last_var_.at(b, n);
            float inv_std = 1.f / std::sqrt(var + eps);

            for (int d = 0; d < D; ++d) {
                float go = grad_out.at(b, n, d);
                dgamma.at(d) += go * last_xhat_.at(b, n, d);
                dbeta.at(d)  += go;
            }

            float sum_go_gamma = 0.f, sum_go_gamma_xh = 0.f;
            for (int d = 0; d < D; ++d) {
                float go_g = grad_out.at(b, n, d) * gamma.at(d);
                sum_go_gamma    += go_g;
                sum_go_gamma_xh += go_g * last_xhat_.at(b, n, d);
            }

            float scale_factor = inv_std / static_cast<float>(D);
            for (int d = 0; d < D; ++d) {
                float go_g = grad_out.at(b, n, d) * gamma.at(d);
                grad_in.at(b, n, d) = scale_factor *
                    (static_cast<float>(D) * go_g
                     - sum_go_gamma
                     - last_xhat_.at(b, n, d) * sum_go_gamma_xh);
            }
        }
    }
    return grad_in;
}

void LayerNorm::zero_grad() { dgamma.zero_(); dbeta.zero_(); }

static inline float gelu_fwd(float x) {
    float s = 1.f / (1.f + std::exp(-1.702f * x));
    return x * s;
}

static inline float gelu_grad(float x) {
    float s = 1.f / (1.f + std::exp(-1.702f * x));
    return s + x * 1.702f * s * (1.f - s);
}

Tensor gelu(const Tensor& x) {
    Tensor out(x.shape);
    for (int i = 0; i < x.numel(); ++i)
        out.data[i] = gelu_fwd(x.data[i]);
    return out;
}

Tensor gelu_backward(const Tensor& x, const Tensor& grad_out) {
    assert(x.data.size() == grad_out.data.size());
    Tensor grad(x.shape);
    for (int i = 0; i < x.numel(); ++i)
        grad.data[i] = gelu_grad(x.data[i]) * grad_out.data[i];
    return grad;
}

Tensor softmax(const Tensor& x) {
    assert(x.ndim() == 2);
    int B = x.shape[0], C = x.shape[1];
    Tensor out(x.shape);
    for (int b = 0; b < B; ++b) {
        float max_v = *std::max_element(&x.data[b * C], &x.data[b * C] + C);
        float sum   = 0.f;
        for (int c = 0; c < C; ++c) {
            out.at(b, c) = std::exp(x.at(b, c) - max_v);
            sum += out.at(b, c);
        }
        float inv_sum = 1.f / sum;
        for (int c = 0; c < C; ++c)
            out.at(b, c) *= inv_sum;
    }
    return out;
}

Tensor softmax_backward(const Tensor& softmax_out, const Tensor& grad_out) {
    assert(softmax_out.ndim() == 2 && grad_out.ndim() == 2);
    int B = softmax_out.shape[0], C = softmax_out.shape[1];
    Tensor grad(softmax_out.shape);
    for (int b = 0; b < B; ++b) {
        float dot = 0.f;
        for (int c = 0; c < C; ++c)
            dot += softmax_out.at(b, c) * grad_out.at(b, c);
        for (int c = 0; c < C; ++c)
            grad.at(b, c) = softmax_out.at(b, c) * (grad_out.at(b, c) - dot);
    }
    return grad;
}

CELossResult cross_entropy(const Tensor& logits, const std::vector<int>& labels) {
    assert(logits.ndim() == 2);
    int B = logits.shape[0], C = logits.shape[1];

    Tensor softmax_out({B, C});
    float total_loss = 0.f;

    for (int b = 0; b < B; ++b) {
        float max_v = *std::max_element(&logits.data[b * C], &logits.data[b * C] + C);
        float sum   = 0.f;
        for (int c = 0; c < C; ++c) {
            float e = std::exp(logits.at(b, c) - max_v);
            softmax_out.at(b, c) = e;
            sum += e;
        }
        for (int c = 0; c < C; ++c)
            softmax_out.at(b, c) /= sum;

        total_loss -= std::log(softmax_out.at(b, labels[b]) + 1e-9f);
    }

    float loss = total_loss / static_cast<float>(B);

    Tensor grad({B, C});
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c)
            grad.at(b, c) = softmax_out.at(b, c) / static_cast<float>(B);
        grad.at(b, labels[b]) -= 1.f / static_cast<float>(B);
    }

    return {loss, std::move(grad)};
}

} // namespace vit
