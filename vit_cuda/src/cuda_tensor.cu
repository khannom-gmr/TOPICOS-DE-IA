#include "cuda_tensor.cuh"
#include <cstdio>

namespace vit::cuda {

__global__ void gelu_kernel(const float* __restrict__ x,
                             float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float s = 1.f / (1.f + expf(-1.702f * v));
    out[i] = v * s;
}

__global__ void gelu_backward_kernel(const float* __restrict__ x,
                                     const float* __restrict__ grad_out,
                                     float*       __restrict__ grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float s = 1.f / (1.f + expf(-1.702f * v));
    grad[i] = (s + v * 1.702f * s * (1.f - s)) * grad_out[i];
}

__global__ void add_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void scale_kernel(float* __restrict__ data, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= s;
}

__global__ void add_bias_3d_kernel(const float* __restrict__ x,
                                   const float* __restrict__ bias,
                                   float*       __restrict__ out,
                                   int B, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;
    if (idx >= total) return;
    int d = idx % D;
    out[idx] = x[idx] + bias[d];
}

__global__ void layer_norm_fwd_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float*       __restrict__ out,
                                      float*       __restrict__ mean,
                                      float*       __restrict__ var,
                                      float*       __restrict__ xhat,
                                      int D, float eps) {
    extern __shared__ float smem[];

    int bn  = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_x = x + bn * D;
    float*       row_o = out   + bn * D;
    float*       row_h = xhat  + bn * D;

    float* s1 = smem;
    float* s2 = smem + blockDim.x;

    float local_sum = 0.f;
    for (int d = tid; d < D; d += blockDim.x)
        local_sum += row_x[d];
    s1[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s1[tid] += s1[tid + stride];
        __syncthreads();
    }
    float mu = s1[0] / static_cast<float>(D);
    if (tid == 0) mean[bn] = mu;
    __syncthreads();

    float local_var = 0.f;
    for (int d = tid; d < D; d += blockDim.x) {
        float diff = row_x[d] - mu;
        local_var += diff * diff;
    }
    s2[tid] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s2[tid] += s2[tid + stride];
        __syncthreads();
    }
    float var_val = s2[0] / static_cast<float>(D);
    if (tid == 0) var[bn] = var_val;
    float inv_std = rsqrtf(var_val + eps);
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float xh = (row_x[d] - mu) * inv_std;
        row_h[d] = xh;
        row_o[d] = gamma[d] * xh + beta[d];
    }
}

__global__ void layer_norm_bwd_kernel(const float* __restrict__ grad_out,
                                      const float* __restrict__ xhat,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ var,
                                      float*       __restrict__ grad_in,
                                      float*       __restrict__ dgamma,
                                      float*       __restrict__ dbeta,
                                      int BN, int D, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BN * D) return;

    int bn = idx / D;
    int d  = idx % D;

    float go = grad_out[bn * D + d];
    float xh = xhat[bn * D + d];

    atomicAdd(&dgamma[d], go * xh);
    atomicAdd(&dbeta[d],  go);

    (void)var; (void)grad_in;
}

__global__ void attn_softmax_fwd_kernel(const float* __restrict__ scores,
                                        float*       __restrict__ out,
                                        int BH, int Nq, int Nk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = BH * Nq;
    if (idx >= total_rows) return;

    const float* row = scores + idx * Nk;
    float*       o   = out    + idx * Nk;

    float max_v = -1e38f;
    for (int j = 0; j < Nk; ++j) max_v = fmaxf(max_v, row[j]);

    float sum = 0.f;
    for (int j = 0; j < Nk; ++j) { o[j] = expf(row[j] - max_v); sum += o[j]; }

    float inv = 1.f / sum;
    for (int j = 0; j < Nk; ++j) o[j] *= inv;
}

__global__ void attn_softmax_bwd_kernel(const float* __restrict__ A,
                                        const float* __restrict__ dA,
                                        float*       __restrict__ dS,
                                        int BH, int Nq, int Nk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = BH * Nq;
    if (idx >= total_rows) return;

    const float* a  = A  + idx * Nk;
    const float* da = dA + idx * Nk;
    float*       ds = dS + idx * Nk;

    float dot = 0.f;
    for (int j = 0; j < Nk; ++j) dot += a[j] * da[j];
    for (int j = 0; j < Nk; ++j) ds[j] = a[j] * (da[j] - dot);
}

__global__ void cross_entropy_fwd_kernel(const float* __restrict__ logits,
                                         const int*   __restrict__ labels,
                                         float*       __restrict__ probs,
                                         float*       __restrict__ losses,
                                         int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* row = logits + b * C;
    float*       p   = probs  + b * C;

    float max_v = -1e38f;
    for (int c = 0; c < C; ++c) max_v = fmaxf(max_v, row[c]);

    float sum = 0.f;
    for (int c = 0; c < C; ++c) { p[c] = expf(row[c] - max_v); sum += p[c]; }
    float inv = 1.f / sum;
    for (int c = 0; c < C; ++c) p[c] *= inv;

    losses[b] = -logf(p[labels[b]] + 1e-9f);
}

__global__ void cross_entropy_bwd_kernel(const float* __restrict__ probs,
                                         const int*   __restrict__ labels,
                                         float*       __restrict__ grad,
                                         int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    int b = idx / C, c = idx % C;
    float g = probs[idx] / static_cast<float>(B);
    if (c == labels[b]) g -= 1.f / static_cast<float>(B);
    grad[idx] = g;
}

__global__ void accum_pos_grad_kernel(const float* __restrict__ d_tokens,
                                      float*       __restrict__ d_pos,
                                      int B, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * D) return;
    int s = idx / D, d = idx % D;
    float acc = 0.f;
    for (int b = 0; b < B; ++b)
        acc += d_tokens[(b * S + s) * D + d];
    atomicAdd(&d_pos[idx], acc);
}

__global__ void accum_cls_grad_kernel(const float* __restrict__ d_tokens,
                                      float*       __restrict__ d_cls,
                                      int B, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int b = 0; b < B; ++b)
        acc += d_tokens[b * D + d];
    atomicAdd(&d_cls[d], acc);
}

void cublas_gemm(cublasHandle_t handle,
                 const CudaTensor& A, const CudaTensor& B,
                 CudaTensor& C, int M, int K, int N,
                 float alpha, float beta) {
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.d_data, N,
                A.d_data, K,
                &beta,
                C.d_data, N);
}

void cublas_gemm_tn(cublasHandle_t handle,
                    const CudaTensor& A, const CudaTensor& B,
                    CudaTensor& C, int M, int K, int N,
                    float alpha, float beta) {
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K,
                &alpha,
                B.d_data, N,
                A.d_data, M,
                &beta,
                C.d_data, N);
}

void cublas_gemm_nt(cublasHandle_t handle,
                    const CudaTensor& A, const CudaTensor& B,
                    CudaTensor& C, int M, int K, int N,
                    float alpha, float beta) {
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.d_data, K,
                A.d_data, K,
                &beta,
                C.d_data, N);
}

void cublas_gemm_batched(cublasHandle_t handle,
                         const CudaTensor& A, const CudaTensor& B,
                         CudaTensor& C, int batch, int M, int K, int N,
                         float alpha, float beta) {
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;

    cublasSgemmStridedBatched(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              B.d_data, N, strideB,
                              A.d_data, K, strideA,
                              &beta,
                              C.d_data, N, strideC,
                              batch);
}

} // namespace vit::cuda
