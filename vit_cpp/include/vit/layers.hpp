#pragma once
#include "vit/tensor.hpp"
#include <vector>

namespace vit {

// -----------------------------------------------------------------------
// Linear Layer: y = x @ W^T + b
//   W: [out, in], b: [out]
// -----------------------------------------------------------------------
class Linear {
public:
    Tensor W, b;    // Parametros
    Tensor dW, db;  // Gradientes acumulados

    Linear() = default;
    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);

    void zero_grad();

    std::vector<Tensor*> parameters()  { return {&W, &b};   }
    std::vector<Tensor*> gradients()   { return {&dW, &db}; }

private:
    Tensor last_input_;
};

// -----------------------------------------------------------------------
// Layer Normalization sobre la ultima dimension (Slide 88/112).
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// -----------------------------------------------------------------------
class LayerNorm {
public:
    Tensor gamma, beta;
    Tensor dgamma, dbeta;
    float  eps;

    LayerNorm() = default;
    explicit LayerNorm(int dim, float eps = 1e-5f);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);

    void zero_grad();

    std::vector<Tensor*> parameters()  { return {&gamma, &beta};   }
    std::vector<Tensor*> gradients()   { return {&dgamma, &dbeta}; }

private:
    Tensor last_x_;
    Tensor last_xhat_;
    Tensor last_var_;
    Tensor last_mean_;
    int    dim_ = 0;
};

// Activation GELU: GeLU(x) = x * sigmoid(1.702 * x)
Tensor gelu(const Tensor& x);
Tensor gelu_backward(const Tensor& x, const Tensor& grad_out);

// Softmax y Cross-Entropy Loss
Tensor softmax(const Tensor& x);
Tensor softmax_backward(const Tensor& softmax_out, const Tensor& grad_out);

struct CELossResult {
    float   loss;
    Tensor  grad;
};

CELossResult cross_entropy(const Tensor& logits, const std::vector<int>& labels);

} // namespace vit
