#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

namespace vit::cuda {

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void gelu_kernel(const float* __restrict__ x,
                             float*       __restrict__ out,
                             int n);

__global__ void gelu_backward_kernel(const float* __restrict__ x,
                                     const float* __restrict__ grad_out,
                                     float*       __restrict__ grad,
                                     int n);

__global__ void add_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ out,
                            int n);

__global__ void scale_kernel(float* __restrict__ data, float s, int n);

__global__ void add_bias_3d_kernel(const float* __restrict__ x,
                                   const float* __restrict__ bias,
                                   float*       __restrict__ out,
                                   int B, int N, int D);

// Layer Normalization (Usa Memoria Compartida / Shared Memory para reduccion rapida de media y varianza)
__global__ void layer_norm_fwd_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float*       __restrict__ out,
                                      float*       __restrict__ mean,
                                      float*       __restrict__ var,
                                      float*       __restrict__ xhat,
                                      int D, float eps);

__global__ void layer_norm_bwd_kernel(const float* __restrict__ grad_out,
                                      const float* __restrict__ xhat,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ var,
                                      float*       __restrict__ grad_in,
                                      float*       __restrict__ dgamma,
                                      float*       __restrict__ dbeta,
                                      int BN, int D, float eps);

// Softmax para Attention (Usa Memoria Compartida / Shared Memory para reduccion de maximo y suma exponencial)
__global__ void attn_softmax_fwd_kernel(const float* __restrict__ scores,
                                        float*       __restrict__ out,
                                        int BH, int Nq, int Nk);

__global__ void attn_softmax_bwd_kernel(const float* __restrict__ A,
                                        const float* __restrict__ dA,
                                        float*       __restrict__ dS,
                                        int BH, int Nq, int Nk);

__global__ void cross_entropy_fwd_kernel(const float* __restrict__ logits,
                                         const int*   __restrict__ labels,
                                         float*       __restrict__ probs,
                                         float*       __restrict__ losses,
                                         int B, int C);

__global__ void cross_entropy_bwd_kernel(const float* __restrict__ probs,
                                         const int*   __restrict__ labels,
                                         float*       __restrict__ grad,
                                         int B, int C);

__global__ void accum_pos_grad_kernel(const float* __restrict__ d_tokens,
                                      float*       __restrict__ d_pos,
                                      int B, int S, int D);

__global__ void accum_cls_grad_kernel(const float* __restrict__ d_tokens,
                                      float*       __restrict__ d_cls,
                                      int B, int D);

} // namespace vit::cuda
