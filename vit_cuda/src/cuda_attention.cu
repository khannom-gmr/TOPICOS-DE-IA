#include "cuda_attention.cuh"
#include <cmath>
#include <vector>
#include <cstring>

namespace vit::cuda {

CudaMultiHeadAttention::CudaMultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim_(embed_dim), num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      W_q(embed_dim, embed_dim),
      W_k(embed_dim, embed_dim),
      W_v(embed_dim, embed_dim),
      W_o(embed_dim, embed_dim) {}

CudaTensor CudaMultiHeadAttention::split_heads_device(const CudaTensor& x_flat,
                                                       int B, int N) const {
    int H = num_heads_, d = head_dim_;
    std::vector<float> h_in(B * N * H * d), h_out(B * H * N * d);
    x_flat.to_host(h_in);

    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int dv = 0; dv < d; ++dv) {
                    int src = (b * N + n) * (H * d) + h * d + dv;
                    int dst = (b * H + h) * N * d + n * d + dv;
                    h_out[dst] = h_in[src];
                }

    CudaTensor out({B * H, N, d});
    out.from_host(h_out);
    return out;
}

CudaTensor CudaMultiHeadAttention::merge_heads_device(const CudaTensor& heads,
                                                       int B, int N) const {
    int H = num_heads_, d = head_dim_;
    std::vector<float> h_in(B * H * N * d), h_out(B * N * H * d);
    heads.to_host(h_in);

    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int n = 0; n < N; ++n)
                for (int dv = 0; dv < d; ++dv) {
                    int src = (b * H + h) * N * d + n * d + dv;
                    int dst = (b * N + n) * H * d + h * d + dv;
                    h_out[dst] = h_in[src];
                }

    CudaTensor out({B * N, H * d});
    out.from_host(h_out);
    return out;
}

CudaTensor CudaMultiHeadAttention::forward(cublasHandle_t handle,
                                            const CudaTensor& x,
                                            int B, int N) {
    last_x_ = CudaTensor({B * N, embed_dim_});
    CUDA_CHECK(cudaMemcpy(last_x_.d_data, x.d_data,
                          B * N * embed_dim_ * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    CudaTensor Q_flat = W_q.forward(handle, x, B * N, embed_dim_, embed_dim_);
    CudaTensor K_flat = W_k.forward(handle, x, B * N, embed_dim_, embed_dim_);
    CudaTensor V_flat = W_v.forward(handle, x, B * N, embed_dim_, embed_dim_);

    last_Q_ = split_heads_device(Q_flat, B, N);
    last_K_ = split_heads_device(K_flat, B, N);
    last_V_ = split_heads_device(V_flat, B, N);

    int BH = B * num_heads_, d = head_dim_;

    CudaTensor scores({BH, N, N});
    float scale = 1.f / std::sqrt(static_cast<float>(d));

    for (int bh = 0; bh < BH; ++bh) {
        float* qptr = last_Q_.d_data + bh * N * d;
        float* kptr = last_K_.d_data + bh * N * d;
        float* sptr = scores.d_data  + bh * N * N;

        CudaTensor Q_slice, K_slice, S_slice;
        Q_slice.d_data = qptr; Q_slice.shape = {N, d};
        K_slice.d_data = kptr; K_slice.shape = {N, d};
        S_slice.d_data = sptr; S_slice.shape = {N, N};
        cublas_gemm_nt(handle, Q_slice, K_slice, S_slice, N, d, N, scale, 0.f);
        Q_slice.d_data = nullptr; K_slice.d_data = nullptr; S_slice.d_data = nullptr;
    }

    last_attn_ = CudaTensor({BH, N, N});
    int attn_rows = BH * N;
    int threads, blocks;
    launch_1d(attn_rows, blocks, threads);
    attn_softmax_fwd_kernel<<<blocks, threads>>>(
        scores.d_data, last_attn_.d_data, BH, N, N);

    last_ctx_heads_ = CudaTensor({BH, N, d});
    for (int bh = 0; bh < BH; ++bh) {
        float* aptr = last_attn_.d_data     + bh * N * N;
        float* vptr = last_V_.d_data        + bh * N * d;
        float* cptr = last_ctx_heads_.d_data + bh * N * d;

        CudaTensor A_sl, V_sl, C_sl;
        A_sl.d_data = aptr; A_sl.shape = {N, N};
        V_sl.d_data = vptr; V_sl.shape = {N, d};
        C_sl.d_data = cptr; C_sl.shape = {N, d};
        cublas_gemm(handle, A_sl, V_sl, C_sl, N, N, d, 1.f, 0.f);
        A_sl.d_data = nullptr; V_sl.d_data = nullptr; C_sl.d_data = nullptr;
    }

    CudaTensor ctx_flat = merge_heads_device(last_ctx_heads_, B, N);

    return W_o.forward(handle, ctx_flat, B * N, embed_dim_, embed_dim_);
}

CudaTensor CudaMultiHeadAttention::backward(cublasHandle_t handle,
                                             const CudaTensor& grad_out,
                                             int B, int N) {
    int BH = B * num_heads_, d = head_dim_;

    CudaTensor d_ctx_flat = W_o.backward(handle, grad_out, B * N, embed_dim_, embed_dim_);

    CudaTensor d_ctx_heads = split_heads_device(d_ctx_flat, B, N);

    CudaTensor d_attn({BH, N, N});
    for (int bh = 0; bh < BH; ++bh) {
        float* dc = d_ctx_heads.d_data + bh * N * d;
        float* vp = last_V_.d_data     + bh * N * d;
        float* da = d_attn.d_data      + bh * N * N;

        CudaTensor DC, VP, DA;
        DC.d_data = dc; DC.shape = {N, d};
        VP.d_data = vp; VP.shape = {N, d};
        DA.d_data = da; DA.shape = {N, N};
        cublas_gemm_nt(handle, DC, VP, DA, N, d, N, 1.f, 0.f);
        DC.d_data = nullptr; VP.d_data = nullptr; DA.d_data = nullptr;
    }

    CudaTensor d_V({BH, N, d});
    for (int bh = 0; bh < BH; ++bh) {
        float* ap = last_attn_.d_data  + bh * N * N;
        float* dc = d_ctx_heads.d_data + bh * N * d;
        float* dv = d_V.d_data         + bh * N * d;

        CudaTensor AP, DC, DV;
        AP.d_data = ap; AP.shape = {N, N};
        DC.d_data = dc; DC.shape = {N, d};
        DV.d_data = dv; DV.shape = {N, d};
        cublas_gemm_tn(handle, AP, DC, DV, N, N, d, 1.f, 0.f);
        AP.d_data = nullptr; DC.d_data = nullptr; DV.d_data = nullptr;
    }

    CudaTensor d_scores({BH, N, N});
    {
        int rows = BH * N, threads, blocks;
        launch_1d(rows, blocks, threads);
        attn_softmax_bwd_kernel<<<blocks, threads>>>(
            last_attn_.d_data, d_attn.d_data, d_scores.d_data, BH, N, N);
    }

    float scale = 1.f / std::sqrt(static_cast<float>(d));
    { int threads, blocks; launch_1d(BH * N * N, blocks, threads);
      scale_kernel<<<blocks, threads>>>(d_scores.d_data, scale, BH * N * N); }

    CudaTensor d_Q({BH, N, d});
    for (int bh = 0; bh < BH; ++bh) {
        float* ds = d_scores.d_data + bh * N * N;
        float* kp = last_K_.d_data  + bh * N * d;
        float* dq = d_Q.d_data      + bh * N * d;

        CudaTensor DS, KP, DQ;
        DS.d_data = ds; DS.shape = {N, N};
        KP.d_data = kp; KP.shape = {N, d};
        DQ.d_data = dq; DQ.shape = {N, d};
        cublas_gemm(handle, DS, KP, DQ, N, N, d, 1.f, 0.f);
        DS.d_data = nullptr; KP.d_data = nullptr; DQ.d_data = nullptr;
    }

    CudaTensor d_K({BH, N, d});
    for (int bh = 0; bh < BH; ++bh) {
        float* ds = d_scores.d_data + bh * N * N;
        float* qp = last_Q_.d_data  + bh * N * d;
        float* dk = d_K.d_data      + bh * N * d;

        CudaTensor DS, QP, DK;
        DS.d_data = ds; DS.shape = {N, N};
        QP.d_data = qp; QP.shape = {N, d};
        DK.d_data = dk; DK.shape = {N, d};
        cublas_gemm_tn(handle, DS, QP, DK, N, N, d, 1.f, 0.f);
        DS.d_data = nullptr; QP.d_data = nullptr; DK.d_data = nullptr;
    }

    CudaTensor d_Q_flat = merge_heads_device(d_Q, B, N);
    CudaTensor d_K_flat = merge_heads_device(d_K, B, N);
    CudaTensor d_V_flat = merge_heads_device(d_V, B, N);

    CudaTensor dx_q = W_q.backward(handle, d_Q_flat, B * N, embed_dim_, embed_dim_);
    CudaTensor dx_k = W_k.backward(handle, d_K_flat, B * N, embed_dim_, embed_dim_);
    CudaTensor dx_v = W_v.backward(handle, d_V_flat, B * N, embed_dim_, embed_dim_);

    CudaTensor dx({B * N, embed_dim_});
    {
        int n = B * N * embed_dim_, threads, blocks;
        launch_1d(n, blocks, threads);
        add_kernel<<<blocks, threads>>>(dx_q.d_data, dx_k.d_data, dx.d_data, n);
        add_kernel<<<blocks, threads>>>(dx.d_data,   dx_v.d_data, dx.d_data, n);
    }

    return dx;
}

void CudaMultiHeadAttention::zero_grad() {
    W_q.zero_grad(); W_k.zero_grad();
    W_v.zero_grad(); W_o.zero_grad();
}

void CudaMultiHeadAttention::get_params(std::vector<CudaTensor*>& params,
                                         std::vector<CudaTensor*>& grads) {
    W_q.get_params(params, grads);
    W_k.get_params(params, grads);
    W_v.get_params(params, grads);
    W_o.get_params(params, grads);
}

} // namespace vit::cuda
