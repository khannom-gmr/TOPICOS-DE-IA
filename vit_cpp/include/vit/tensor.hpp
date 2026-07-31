#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace vit {

// -----------------------------------------------------------------------
// Tensor - Array multidimensional (hasta 4D) almacenado en row-major.
// Manejo explicito de gradientes en cada capa (sin autograd dinamico).
// -----------------------------------------------------------------------
class Tensor {
public:
    std::vector<float> data;
    std::vector<int>   shape;

    Tensor() = default;
    explicit Tensor(std::vector<int> shape)
        : shape(shape), data(total_elems(shape), 0.f) {}

    explicit Tensor(std::vector<int> shape, float fill_val)
        : shape(shape), data(total_elems(shape), fill_val) {}

    int ndim()  const { return static_cast<int>(shape.size()); }
    int numel() const { return static_cast<int>(data.size()); }
    int dim(int d) const { return shape.at(d); }

    // Acceso por indices (row-major)
    float& at(int i)                     { return data[i]; }
    const float& at(int i)         const { return data[i]; }

    float& at(int i, int j)              { return data[i * shape[1] + j]; }
    const float& at(int i, int j)  const { return data[i * shape[1] + j]; }

    float& at(int b, int i, int j) {
        return data[(b * shape[1] + i) * shape[2] + j];
    }
    const float& at(int b, int i, int j) const {
        return data[(b * shape[1] + i) * shape[2] + j];
    }

    float& at(int b, int h, int i, int j) {
        return data[((b * shape[1] + h) * shape[2] + i) * shape[3] + j];
    }
    const float& at(int b, int h, int i, int j) const {
        return data[((b * shape[1] + h) * shape[2] + i) * shape[3] + j];
    }

    void zero_()                  { std::fill(data.begin(), data.end(), 0.f); }
    void fill_(float v)           { std::fill(data.begin(), data.end(), v);   }
    void add_inplace_(const Tensor& other);
    void scale_inplace_(float s);

    Tensor reshape(std::vector<int> new_shape) const;
    Tensor transpose_2d()    const; // [M,N] -> [N,M]
    Tensor transpose_3d_12() const; // [B,M,N] -> [B,N,M]
    Tensor transpose_4d_23() const; // [B,H,M,N] -> [B,H,N,M]

    void randn_(float std = 1.f);
    void xavier_uniform_(int fan_in, int fan_out);
    void kaiming_uniform_(int fan_in);

    std::string shape_str() const;
    float sum() const { return std::accumulate(data.begin(), data.end(), 0.f); }
    float mean() const { return sum() / static_cast<float>(numel()); }

private:
    static int total_elems(const std::vector<int>& sh) {
        int n = 1;
        for (int d : sh) n *= d;
        return n;
    }
};

// Operaciones matriciales y de tensores
Tensor matmul(const Tensor& A, const Tensor& B);
Tensor bmm(const Tensor& A, const Tensor& B);
Tensor add(const Tensor& A, const Tensor& B);
Tensor add_bias_3d(const Tensor& x, const Tensor& b);
Tensor mul(const Tensor& A, const Tensor& B);
Tensor scale(const Tensor& A, float s);
void clip_inplace(Tensor& t, float max_val);

} // namespace vit
