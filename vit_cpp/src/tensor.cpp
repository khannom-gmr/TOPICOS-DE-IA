#include "vit/tensor.hpp"
#include <random>
#include <stdexcept>
#include <sstream>
#include <cmath>

namespace vit {

void Tensor::add_inplace_(const Tensor& other) {
    assert(data.size() == other.data.size());
    for (size_t i = 0; i < data.size(); ++i)
        data[i] += other.data[i];
}

void Tensor::scale_inplace_(float s) {
    for (auto& v : data) v *= s;
}

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    Tensor out;
    out.data  = data;
    out.shape = new_shape;
    return out;
}

Tensor Tensor::transpose_2d() const {
    assert(ndim() == 2);
    int M = shape[0], N = shape[1];
    Tensor out({N, M});
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            out.at(j, i) = at(i, j);
    return out;
}

Tensor Tensor::transpose_3d_12() const {
    assert(ndim() == 3);
    int B = shape[0], M = shape[1], N = shape[2];
    Tensor out({B, N, M});
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                out.at(b, j, i) = at(b, i, j);
    return out;
}

Tensor Tensor::transpose_4d_23() const {
    assert(ndim() == 4);
    int B = shape[0], H = shape[1], M = shape[2], N = shape[3];
    Tensor out({B, H, N, M});
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    out.at(b, h, j, i) = at(b, h, i, j);
    return out;
}

void Tensor::randn_(float std) {
    static std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, std);
    for (auto& v : data) v = dist(rng);
}

void Tensor::xavier_uniform_(int fan_in, int fan_out) {
    float limit = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& v : data) v = dist(rng);
}

void Tensor::kaiming_uniform_(int fan_in) {
    float std = std::sqrt(2.f / static_cast<float>(fan_in));
    randn_(std);
}

std::string Tensor::shape_str() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 2 && B.ndim() == 2);
    int M = A.shape[0], K = A.shape[1], N = B.shape[1];
    assert(K == B.shape[0]);

    Tensor C({M, N});
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float a = A.at(i, k);
            if (a == 0.f) continue;
            for (int j = 0; j < N; ++j)
                C.at(i, j) += a * B.at(k, j);
        }
    return C;
}

Tensor bmm(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 3 && B.ndim() == 3);
    int batch = A.shape[0], M = A.shape[1], K = A.shape[2];
    int N = B.shape[2];
    assert(K == B.shape[1] && batch == B.shape[0]);

    Tensor C({batch, M, N});
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float a = A.at(b, i, k);
                if (a == 0.f) continue;
                for (int j = 0; j < N; ++j)
                    C.at(b, i, j) += a * B.at(b, k, j);
            }
    return C;
}

Tensor add(const Tensor& A, const Tensor& B) {
    assert(A.data.size() == B.data.size());
    Tensor C(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

Tensor add_bias_3d(const Tensor& x, const Tensor& b) {
    assert(x.ndim() == 3);
    int B = x.shape[0], N = x.shape[1], D = x.shape[2];
    assert(static_cast<int>(b.data.size()) == D);
    Tensor out(x.shape);
    for (int bi = 0; bi < B; ++bi)
        for (int n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
                out.at(bi, n, d) = x.at(bi, n, d) + b.at(d);
    return out;
}

Tensor mul(const Tensor& A, const Tensor& B) {
    assert(A.data.size() == B.data.size());
    Tensor C(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i)
        C.data[i] = A.data[i] * B.data[i];
    return C;
}

Tensor scale(const Tensor& A, float s) {
    Tensor C(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i)
        C.data[i] = A.data[i] * s;
    return C;
}

void clip_inplace(Tensor& t, float max_val) {
    for (auto& v : t.data)
        v = std::max(-max_val, std::min(max_val, v));
}

} // namespace vit
