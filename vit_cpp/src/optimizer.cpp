#include "vit/optimizer.hpp"
#include <cassert>
#include <cmath>
#include <numeric>

namespace vit {

AdamOptimizer::AdamOptimizer(std::vector<Tensor*> params,
                             std::vector<Tensor*> grads,
                             Config cfg)
    : cfg_(cfg), params_(params), grads_(grads) {
    assert(params.size() == grads.size());
    m_.reserve(params.size());
    v_.reserve(params.size());
    for (auto* p : params) {
        m_.emplace_back(p->shape, 0.f);
        v_.emplace_back(p->shape, 0.f);
    }
}

void AdamOptimizer::zero_grad() {
    for (auto* g : grads_) g->zero_();
}

void AdamOptimizer::clip_global_norm() {
    if (cfg_.grad_clip <= 0.f) return;

    double sq_norm = 0.0;
    for (auto* g : grads_)
        for (float v : g->data)
            sq_norm += static_cast<double>(v) * v;

    float global_norm = std::sqrt(static_cast<float>(sq_norm));
    if (global_norm > cfg_.grad_clip) {
        float scale = cfg_.grad_clip / (global_norm + 1e-6f);
        for (auto* g : grads_)
            for (auto& v : g->data)
                v *= scale;
    }
}

void AdamOptimizer::step() {
    ++step_;
    clip_global_norm();

    float lr    = cfg_.lr;
    float beta1 = cfg_.beta1, beta2 = cfg_.beta2, eps = cfg_.eps;
    float wd    = cfg_.weight_decay;

    float bc1 = 1.f - std::pow(beta1, static_cast<float>(step_));
    float bc2 = 1.f - std::pow(beta2, static_cast<float>(step_));
    float lr_t = lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& p  = *params_[i];
        Tensor& g  = *grads_[i];
        Tensor& m  =  m_[i];
        Tensor& v  =  v_[i];

        for (int j = 0; j < p.numel(); ++j) {
            float gj = g.data[j];

            p.data[j] -= lr * wd * p.data[j];

            m.data[j] = beta1 * m.data[j] + (1.f - beta1) * gj;
            v.data[j] = beta2 * v.data[j] + (1.f - beta2) * gj * gj;

            p.data[j] -= lr_t * m.data[j] / (std::sqrt(v.data[j]) + eps);
        }
    }
}

void AdamOptimizer::cosine_schedule(int epoch, int total_epochs, float lr_min) {
    float progress = static_cast<float>(epoch) / static_cast<float>(total_epochs);
    float factor   = 0.5f * (1.f + std::cos(3.14159265f * progress));
    cfg_.lr = lr_min + (cfg_.lr - lr_min) * factor;
}

} // namespace vit
