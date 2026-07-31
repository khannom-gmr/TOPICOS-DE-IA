#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include "kernels.cuh"

namespace vit::cuda {

class CudaTensor {
public:
    float*           d_data = nullptr;
    std::vector<int> shape;

    CudaTensor() = default;

    explicit CudaTensor(std::vector<int> sh)
        : shape(sh) { allocate(); }

    CudaTensor(std::vector<int> sh, float fill_val)
        : shape(sh) { allocate(); fill_(fill_val); }

    CudaTensor(const CudaTensor&)            = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    CudaTensor(CudaTensor&& o) noexcept
        : d_data(o.d_data), shape(std::move(o.shape)) { o.d_data = nullptr; }

    CudaTensor& operator=(CudaTensor&& o) noexcept {
        if (this != &o) { free_(); d_data = o.d_data; shape = std::move(o.shape); o.d_data = nullptr; }
        return *this;
    }

    ~CudaTensor() { free_(); }

    int ndim()  const { return static_cast<int>(shape.size()); }
    int numel() const { int n = 1; for (int d : shape) n *= d; return n; }

    void from_host(const std::vector<float>& h) {
        assert(static_cast<int>(h.size()) == numel());
        CUDA_CHECK(cudaMemcpy(d_data, h.data(), numel() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void to_host(std::vector<float>& h) const {
        h.resize(numel());
        CUDA_CHECK(cudaMemcpy(h.data(), d_data, numel() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    void zero_() {
        CUDA_CHECK(cudaMemset(d_data, 0, numel() * sizeof(float)));
    }

    void fill_(float val) {
        std::vector<float> h(numel(), val);
        from_host(h);
    }

    void resize(std::vector<int> new_shape) {
        if (new_shape == shape) return;
        free_();
        shape = new_shape;
        allocate();
    }

    std::string shape_str() const {
        std::string s = "[";
        for (int i = 0; i < (int)shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < (int)shape.size()) s += ", ";
        }
        return s + "]";
    }

private:
    void allocate() {
        int n = numel();
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
        zero_();
    }

    void free_() {
        if (d_data) { cudaFree(d_data); d_data = nullptr; }
    }
};

inline void launch_1d(int n, int& blocks, int& threads) {
    threads = 256;
    blocks  = (n + threads - 1) / threads;
}

void cublas_gemm(cublasHandle_t handle,
                 const CudaTensor& A, const CudaTensor& B,
                 CudaTensor& C,
                 int M, int K, int N,
                 float alpha = 1.f, float beta = 0.f);

void cublas_gemm_batched(cublasHandle_t handle,
                         const CudaTensor& A, const CudaTensor& B,
                         CudaTensor& C,
                         int batch, int M, int K, int N,
                         float alpha = 1.f, float beta = 0.f);

void cublas_gemm_tn(cublasHandle_t handle,
                    const CudaTensor& A, const CudaTensor& B,
                    CudaTensor& C,
                    int M, int K, int N,
                    float alpha = 1.f, float beta = 0.f);

void cublas_gemm_nt(cublasHandle_t handle,
                    const CudaTensor& A, const CudaTensor& B,
                    CudaTensor& C,
                    int M, int K, int N,
                    float alpha = 1.f, float beta = 0.f);

} // namespace vit::cuda
