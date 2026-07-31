#pragma once
#include <vector>
#include <chrono>
#include <string>

namespace vit::utils {

struct EpochStats {
    float avg_loss = 0.f;
    float accuracy = 0.f;
    double elapsed_s = 0.0;
    int total_samples = 0;
};

class MetricsTracker {
public:
    void reset();
    void update(float loss, int correct, int batch_size);
    EpochStats compute(double elapsed_s) const;

    float running_loss() const;
    float running_accuracy() const;

private:
    double loss_sum_ = 0.0;
    int correct_ = 0;
    int total_ = 0;
    int batches_ = 0;
};

int count_correct(const std::vector<float>& logits,
                  const std::vector<int>&   labels,
                  int batch_size, int num_classes);

} // namespace vit::utils
