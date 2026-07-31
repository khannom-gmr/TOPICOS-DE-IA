#include "utils/metrics.hpp"
#include <algorithm>
#include <stdexcept>

namespace vit::utils {

void MetricsTracker::reset() {
    loss_sum_ = 0.0;
    correct_  = 0;
    total_    = 0;
    batches_  = 0;
}

void MetricsTracker::update(float loss, int correct, int batch_size) {
    loss_sum_ += static_cast<double>(loss) * batch_size;
    correct_  += correct;
    total_    += batch_size;
    ++batches_;
}

EpochStats MetricsTracker::compute(double elapsed_s) const {
    if (total_ == 0) return {};
    EpochStats s;
    s.avg_loss     = static_cast<float>(loss_sum_ / total_);
    s.accuracy     = static_cast<float>(correct_) / static_cast<float>(total_);
    s.elapsed_s    = elapsed_s;
    s.total_samples = total_;
    return s;
}

float MetricsTracker::running_loss() const {
    if (total_ == 0) return 0.f;
    return static_cast<float>(loss_sum_ / total_);
}

float MetricsTracker::running_accuracy() const {
    if (total_ == 0) return 0.f;
    return static_cast<float>(correct_) / static_cast<float>(total_);
}

int count_correct(const std::vector<float>& logits,
                  const std::vector<int>&   labels,
                  int batch_size, int num_classes) {
    int correct = 0;
    for (int b = 0; b < batch_size; ++b) {
        const float* row = logits.data() + b * num_classes;
        int pred = static_cast<int>(
            std::max_element(row, row + num_classes) - row);
        if (pred == labels[b]) ++correct;
    }
    return correct;
}

} // namespace vit::utils
