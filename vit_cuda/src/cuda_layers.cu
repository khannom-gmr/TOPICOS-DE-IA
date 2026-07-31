#include "cuda_layers.cuh"
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

namespace vit::cuda {

CudaLinear::CudaLinear(int in_features, int out_features)
    : W({out_features, in_features}),
      b({out_features}, 0.f),
      dW({out_features, in_features}, 0.f),
      db({out_features}, 0.f) {
    float std = std::sqrt(2.f / static_cast<float>(in_features));
    std::vector<float> h_W(out_features * in_features);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, std);
    for (auto& v : h_W) v = dist(rng);
    W.from_host(h_W);
}

CudaTensor CudaLinear::forward(cublasHandle_t handle, const CudaTensor& x,
                                int B, int in, int out) {
    last_x_ = CudaTensor({B, in});
    CUDA_CHECK(cudaMemcpy(last_x_.d_data, x.d_data,
                          B * in * sizeof(float), cudaMemcpyDeviceToDevice));

    CudaTensor result({B, out});

    cublas_gemm_nt(handle, x, W, result, B, in, out);

    int threads, blocks;
    launch_1d(B * out, blocks, threads);
    add_bias_3d_kernel<<<blocks, threads>>>(
        result.d_data, b.d_data, result.d_data, B, 1, out);

    return result;
}

CudaTensor CudaLinear::backward(cublasHandle_t handle,
                                 const CudaTensor& grad_out,
                                 int B, int in, int out) {
    cublas_gemm_tn(handle, grad_out, last_x_, dW, out, B, in, 1.f, 1.f);

    std::vector<float> h_go(B * out), h_db(out);
    grad_out.to_host(h_go);
    db.to_host(h_db);
    for (int b = 0; b < B; ++b)
        for (int o = 0; o < out; ++o)
            h_db[o] += h_go[b * out + o];
    db.from_host(h_db);

    CudaTensor grad_in({B, in});
    cublas_gemm(handle, grad_out, W, grad_in, B, out, in);

    return grad_in;
}

void CudaLinear::zero_grad() { dW.zero_(); db.zero_(); }

void CudaLinear::get_params(std::vector<CudaTensor*>& params,
                             std::vector<CudaTensor*>& grads) {
    params.push_back(&W); grads.push_back(&dW);
    params.push_back(&b); grads.push_back(&db);
}

CudaLayerNorm::CudaLayerNorm(int dim, float eps_val)
    : eps(eps_val), dim_(dim),
      gamma({dim}, 1.f),
      beta ({dim}, 0.f),
      dgamma({dim}, 0.f),
      dbeta ({dim}, 0.f) {}

CudaTensor CudaLayerNorm::forward(const CudaTensor& x, int BN, int D) {
    last_BN_   = BN;
    last_xhat_ = CudaTensor({BN, D});
    last_var_  = CudaTensor({BN});
    last_mean_ = CudaTensor({BN});

    CudaTensor out({BN, D});

    int threads = std::min(128, D);
    int smem = 2 * threads * sizeof(float);
    layer_norm_fwd_kernel<<<BN, threads, smem>>>(
        x.d_data, gamma.d_data, beta.d_data,
        out.d_data, last_mean_.d_data, last_var_.d_data, last_xhat_.d_data,
        D, eps);

    return out;
}

CudaTensor CudaLayerNorm::backward(const CudaTensor& grad_out, int BN, int D) {
    int threads, blocks;
    launch_1d(BN * D, blocks, threads);
    layer_norm_bwd_kernel<<<blocks, threads>>>(
        grad_out.d_data, last_xhat_.d_data, gamma.d_data, last_var_.d_data,
        nullptr,
        dgamma.d_data, dbeta.d_data,
        BN, D, eps);

    std::vector<float> h_go(BN * D), h_xhat(BN * D), h_var(BN), h_gamma(D);
    grad_out.to_host(h_go);
    last_xhat_.to_host(h_xhat);
    last_var_.to_host(h_var);
    gamma.to_host(h_gamma);

    std::vector<float> h_gin(BN * D);
    for (int bn = 0; bn < BN; ++bn) {
        float var_val = h_var[bn];
        float inv_std = 1.f / std::sqrt(var_val + eps);
        float sg = 0.f, sgx = 0.f;
        for (int d = 0; d < D; ++d) {
            float go_g = h_go[bn * D + d] * h_gamma[d];
            sg  += go_g;
            sgx += go_g * h_xhat[bn * D + d];
        }
        float factor = inv_std / static_cast<float>(D);
        for (int d = 0; d < D; ++d) {
            float go_g = h_go[bn * D + d] * h_gamma[d];
            h_gin[bn * D + d] = factor *
                (static_cast<float>(D) * go_g - sg - h_xhat[bn * D + d] * sgx);
        }
    }

    CudaTensor grad_in({BN, D});
    grad_in.from_host(h_gin);
    return grad_in;
}

void CudaLayerNorm::zero_grad() { dgamma.zero_(); dbeta.zero_(); }

void CudaLayerNorm::get_params(std::vector<CudaTensor*>& params,
                                std::vector<CudaTensor*>& grads) {
    params.push_back(&gamma); grads.push_back(&dgamma);
    params.push_back(&beta);  grads.push_back(&dbeta);
}

} // namespace vit::cuda
