#pragma once
#include "vit/tensor.hpp"
#include <vector>
#include <cmath>

namespace vit {

// Optimizador AdamW con Weight Decay desacoplado y Cosine Annealing
class AdamOptimizer {
public:
    struct Config {
        float lr           = 1e-3f;
        float beta1        = 0.9f;
        float beta2        = 0.999f;
        float eps          = 1e-8f;
        float weight_decay = 1e-4f;
        float grad_clip    = 1.0f;
    };

    AdamOptimizer() = default;
    AdamOptimizer(std::vector<Tensor*> params,
                  std::vector<Tensor*> grads,
                  Config cfg = {});

    void step();
    void zero_grad();

    void set_lr(float lr) { cfg_.lr = lr; }
    int  current_step()   const { return step_; }

    void cosine_schedule(int epoch, int total_epochs, float lr_min = 1e-5f);

private:
    Config              cfg_;
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
    std::vector<Tensor>  m_;
    std::vector<Tensor>  v_;
    int                  step_ = 0;

    void clip_global_norm();
};

} // namespace vit
