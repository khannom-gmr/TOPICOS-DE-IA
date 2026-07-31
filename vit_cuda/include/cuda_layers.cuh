#pragma once
#include "cuda_tensor.cuh"
#include <cublas_v2.h>
#include <vector>

namespace vit::cuda {

class CudaLinear {
public:
    CudaTensor W, b;
    CudaTensor dW, db;

    CudaLinear() = default;
    CudaLinear(int in_features, int out_features);

    CudaTensor forward(cublasHandle_t handle, const CudaTensor& x,
                       int B, int in, int out);

    CudaTensor backward(cublasHandle_t handle, const CudaTensor& grad_out,
                        int B, int in, int out);

    void zero_grad();

    void get_params(std::vector<CudaTensor*>& params,
                    std::vector<CudaTensor*>& grads);

private:
    CudaTensor last_x_;
};

class CudaLayerNorm {
public:
    CudaTensor gamma, beta;
    CudaTensor dgamma, dbeta;
    float      eps;
    int        dim_;

    CudaLayerNorm() = default;
    explicit CudaLayerNorm(int dim, float eps = 1e-5f);

    CudaTensor forward(const CudaTensor& x, int BN, int D);
    CudaTensor backward(const CudaTensor& grad_out, int BN, int D);

    void zero_grad();

    void get_params(std::vector<CudaTensor*>& params,
                    std::vector<CudaTensor*>& grads);

private:
    CudaTensor last_xhat_;
    CudaTensor last_var_;
    CudaTensor last_mean_;
    int        last_BN_ = 0;
};

} // namespace vit::cuda
