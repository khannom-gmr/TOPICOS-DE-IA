#include "cuda_transformer.cuh"
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>

namespace vit::cuda {

CudaTransformerBlock::CudaTransformerBlock(const vit::ViTConfig& cfg)
    : cfg_(cfg),
      ln1_(cfg.embed_dim),
      ln2_(cfg.embed_dim),
      mhsa_(cfg.embed_dim, cfg.num_heads),
      ff1_(cfg.embed_dim, cfg.mlp_dim),
      ff2_(cfg.mlp_dim, cfg.embed_dim) {}

CudaTensor CudaTransformerBlock::forward(cublasHandle_t handle,
                                           const CudaTensor& x, int B, int S) {
    int D  = cfg_.embed_dim;
    int BN = B * S;

    last_x_ = CudaTensor({BN, D});
    CUDA_CHECK(cudaMemcpy(last_x_.d_data, x.d_data,
                          BN * D * sizeof(float), cudaMemcpyDeviceToDevice));

    last_ln1_out_  = ln1_.forward(x, BN, D);
    last_attn_out_ = mhsa_.forward(handle, last_ln1_out_, B, S);

    last_res1_ = CudaTensor({BN, D});
    {
        int n = BN * D, threads, blocks;
        launch_1d(n, blocks, threads);
        add_kernel<<<blocks, threads>>>(x.d_data, last_attn_out_.d_data,
                                        last_res1_.d_data, n);
    }

    last_ln2_out_       = ln2_.forward(last_res1_, BN, D);
    last_ff1_pre_gelu_  = ff1_.forward(handle, last_ln2_out_, BN, D, cfg_.mlp_dim);
    last_ff1_out_       = CudaTensor({BN, cfg_.mlp_dim});
    {
        int n = BN * cfg_.mlp_dim, threads, blocks;
        launch_1d(n, blocks, threads);
        gelu_kernel<<<blocks, threads>>>(last_ff1_pre_gelu_.d_data,
                                          last_ff1_out_.d_data, n);
    }
    CudaTensor ff2_out = ff2_.forward(handle, last_ff1_out_, BN, cfg_.mlp_dim, D);

    CudaTensor out({BN, D});
    {
        int n = BN * D, threads, blocks;
        launch_1d(n, blocks, threads);
        add_kernel<<<blocks, threads>>>(last_res1_.d_data, ff2_out.d_data,
                                        out.d_data, n);
    }
    return out;
}

CudaTensor CudaTransformerBlock::backward(cublasHandle_t handle,
                                            const CudaTensor& g, int B, int S) {
    int D  = cfg_.embed_dim;
    int BN = B * S;

    CudaTensor d_ff2    = ff2_.backward(handle, g, BN, cfg_.mlp_dim, D);
    CudaTensor d_gelu({BN, cfg_.mlp_dim});
    {
        int n = BN * cfg_.mlp_dim, threads, blocks;
        launch_1d(n, blocks, threads);
        gelu_backward_kernel<<<blocks, threads>>>(
            last_ff1_pre_gelu_.d_data, d_ff2.d_data, d_gelu.d_data, n);
    }
    CudaTensor d_ln2_in = ff1_.backward(handle, d_gelu, BN, D, cfg_.mlp_dim);
    CudaTensor d_res1   = ln2_.backward(d_ln2_in, BN, D);

    {
        int n = BN * D, threads, blocks;
        launch_1d(n, blocks, threads);
        add_kernel<<<blocks, threads>>>(d_res1.d_data, g.d_data,
                                        d_res1.d_data, n);
    }

    CudaTensor d_attn = mhsa_.backward(handle, d_res1, B, S);
    CudaTensor d_ln1  = ln1_.backward(d_attn, BN, D);

    {
        int n = BN * D, threads, blocks;
        launch_1d(n, blocks, threads);
        add_kernel<<<blocks, threads>>>(d_ln1.d_data, d_res1.d_data,
                                        d_ln1.d_data, n);
    }
    return d_ln1;
}

void CudaTransformerBlock::zero_grad() {
    ln1_.zero_grad(); ln2_.zero_grad();
    mhsa_.zero_grad();
    ff1_.zero_grad(); ff2_.zero_grad();
}

void CudaTransformerBlock::get_params(std::vector<CudaTensor*>& params,
                                       std::vector<CudaTensor*>& grads) {
    mhsa_.get_params(params, grads);
    ln1_.get_params(params, grads);
    ln2_.get_params(params, grads);
    ff1_.get_params(params, grads);
    ff2_.get_params(params, grads);
}

CudaViT::CudaViT(const vit::ViTConfig& cfg) : cfg_(cfg) {
    make_cublas();

    patch_proj_  = CudaLinear(cfg.patch_dim(), cfg.embed_dim);
    cls_token_   = CudaTensor({1, cfg.embed_dim});
    pos_embed_   = CudaTensor({cfg.seq_len(), cfg.embed_dim});
    d_cls_token_ = CudaTensor({1, cfg.embed_dim}, 0.f);
    d_pos_embed_ = CudaTensor({cfg.seq_len(), cfg.embed_dim}, 0.f);

    {
        std::mt19937 rng(43);
        std::normal_distribution<float> dist(0.f, 0.02f);

        std::vector<float> h(cfg.embed_dim);
        for (auto& v : h) v = dist(rng);
        cls_token_.from_host(h);

        h.resize(cfg.seq_len() * cfg.embed_dim);
        for (auto& v : h) v = dist(rng);
        pos_embed_.from_host(h);
    }

    blocks_.reserve(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; ++i)
        blocks_.emplace_back(cfg);

    final_norm_ = CudaLayerNorm(cfg.embed_dim);
    classifier_ = CudaLinear(cfg.embed_dim, cfg.num_classes);
}

CudaViT::~CudaViT() {
    if (cublas_) cublasDestroy(cublas_);
}

void CudaViT::make_cublas() {
    cublasStatus_t st = cublasCreate(&cublas_);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasCreate fallo");
}

CudaTensor CudaViT::extract_patches(const std::vector<float>& x_host,
                                     int B, const vit::ViTConfig& cfg) {
    int N  = cfg.num_patches();
    int PD = cfg.patch_dim();
    int ph = cfg.patch_size, pw = cfg.patch_size;
    int iw = cfg.image_size;
    int nh = iw / ph, nw = iw / pw;

    std::vector<float> patches(B * N * PD);
    for (int b = 0; b < B; ++b) {
        const float* img = x_host.data() + b * iw * iw;
        for (int py = 0; py < nh; ++py) {
            for (int px = 0; px < nw; ++px) {
                int pid = py * nw + px;
                for (int dy = 0; dy < ph; ++dy)
                    for (int dx = 0; dx < pw; ++dx)
                        patches[(b * N + pid) * PD + dy * pw + dx] =
                            img[(py * ph + dy) * iw + (px * pw + dx)];
            }
        }
    }

    CudaTensor out({B * N, PD});
    out.from_host(patches);
    return out;
}

CudaTensor CudaViT::forward(const std::vector<float>& x_host, int B) {
    last_B_ = B;
    int D = cfg_.embed_dim;
    int N = cfg_.num_patches();
    int S = cfg_.seq_len();

    CudaTensor patches = extract_patches(x_host, B, cfg_);
    last_x_device_ = patch_proj_.forward(cublas_, patches, B * N,
                                          cfg_.patch_dim(), D);

    std::vector<float> h_cls(D), h_pos(S * D), h_emb(B * N * D),
                       h_tokens(B * S * D);
    cls_token_.to_host(h_cls);
    pos_embed_.to_host(h_pos);
    last_x_device_.to_host(h_emb);

    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d)
            h_tokens[(b * S + 0) * D + d] = h_cls[d] + h_pos[d];
        for (int n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
                h_tokens[(b * S + n + 1) * D + d] =
                    h_emb[(b * N + n) * D + d] + h_pos[(n + 1) * D + d];
    }

    last_tokens_ = CudaTensor({B * S, D});
    last_tokens_.from_host(h_tokens);

    CudaTensor h({B * S, D});
    CUDA_CHECK(cudaMemcpy(h.d_data, last_tokens_.d_data,
                          B * S * D * sizeof(float), cudaMemcpyDeviceToDevice));
    for (auto& block : blocks_)
        h = block.forward(cublas_, h, B, S);

    last_blocks_out_ = CudaTensor({B * S, D});
    CUDA_CHECK(cudaMemcpy(last_blocks_out_.d_data, h.d_data,
                          B * S * D * sizeof(float), cudaMemcpyDeviceToDevice));

    CudaTensor normed = final_norm_.forward(h, B * S, D);

    std::vector<float> h_normed(B * S * D), h_cls_out(B * D);
    normed.to_host(h_normed);
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
            h_cls_out[b * D + d] = h_normed[b * S * D + d];

    last_cls_ = CudaTensor({B, D});
    last_cls_.from_host(h_cls_out);

    return classifier_.forward(cublas_, last_cls_, B, D, cfg_.num_classes);
}

void CudaViT::backward(const CudaTensor& grad_logits, int B) {
    int D = cfg_.embed_dim;
    int N = cfg_.num_patches();
    int S = cfg_.seq_len();

    CudaTensor d_cls = classifier_.backward(cublas_, grad_logits, B, D,
                                            cfg_.num_classes);

    std::vector<float> h_d_normed(B * S * D, 0.f), h_d_cls(B * D);
    d_cls.to_host(h_d_cls);
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
            h_d_normed[b * S * D + d] = h_d_cls[b * D + d];

    CudaTensor d_normed({B * S, D});
    d_normed.from_host(h_d_normed);

    CudaTensor d_h = final_norm_.backward(d_normed, B * S, D);

    CudaTensor g = std::move(d_h);
    for (int i = (int)blocks_.size() - 1; i >= 0; --i)
        g = blocks_[i].backward(cublas_, g, B, S);

    {
        std::vector<float> h_g(B * S * D), h_d_pos(S * D);
        g.to_host(h_g);
        d_pos_embed_.to_host(h_d_pos);
        for (int b = 0; b < B; ++b)
            for (int s = 0; s < S; ++s)
                for (int d = 0; d < D; ++d)
                    h_d_pos[s * D + d] += h_g[(b * S + s) * D + d];
        d_pos_embed_.from_host(h_d_pos);
    }

    {
        std::vector<float> h_g(B * S * D), h_d_cls2(D);
        g.to_host(h_g);
        d_cls_token_.to_host(h_d_cls2);
        for (int b = 0; b < B; ++b)
            for (int d = 0; d < D; ++d)
                h_d_cls2[d] += h_g[b * S * D + d];
        d_cls_token_.from_host(h_d_cls2);
    }

    CudaTensor d_patches({B * N, D});
    {
        std::vector<float> h_g(B * S * D), h_dp(B * N * D, 0.f);
        g.to_host(h_g);
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n)
                for (int d = 0; d < D; ++d)
                    h_dp[(b * N + n) * D + d] = h_g[(b * S + n + 1) * D + d];
        d_patches.from_host(h_dp);
    }
    patch_proj_.backward(cublas_, d_patches, B * N, cfg_.patch_dim(), D);
}

void CudaViT::zero_grad() {
    patch_proj_.zero_grad();
    d_cls_token_.zero_();
    d_pos_embed_.zero_();
    for (auto& block : blocks_) block.zero_grad();
    final_norm_.zero_grad();
    classifier_.zero_grad();
}

std::pair<float, int> CudaViT::loss_and_accuracy(
    const CudaTensor& logits, const std::vector<int>& labels, int B) {

    int C = cfg_.num_classes;

    int* d_labels;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_labels, labels.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    CudaTensor probs({B, C}), losses({B});
    {
        int threads, blocks; launch_1d(B, blocks, threads);
        cross_entropy_fwd_kernel<<<blocks, threads>>>(
            logits.d_data, d_labels, probs.d_data, losses.d_data, B, C);
    }

    std::vector<float> h_losses(B), h_probs(B * C);
    losses.to_host(h_losses);
    probs.to_host(h_probs);

    float total_loss = 0.f;
    for (float l : h_losses) total_loss += l;
    total_loss /= B;

    int correct = 0;
    for (int b = 0; b < B; ++b) {
        int pred = (int)(std::max_element(&h_probs[b * C], &h_probs[b * C] + C)
                         - &h_probs[b * C]);
        if (pred == labels[b]) ++correct;
    }

    cudaFree(d_labels);
    return {total_loss, correct};
}

std::vector<CudaTensor*> CudaViT::parameters() {
    if (!registry_built_) build_registry();
    return all_params_;
}

std::vector<CudaTensor*> CudaViT::gradients() {
    if (!registry_built_) build_registry();
    return all_grads_;
}

void CudaViT::build_registry() {
    all_params_.clear(); all_grads_.clear();
    patch_proj_.get_params(all_params_, all_grads_);
    all_params_.push_back(&cls_token_); all_grads_.push_back(&d_cls_token_);
    all_params_.push_back(&pos_embed_); all_grads_.push_back(&d_pos_embed_);
    for (auto& b : blocks_) b.get_params(all_params_, all_grads_);
    final_norm_.get_params(all_params_, all_grads_);
    classifier_.get_params(all_params_, all_grads_);
    registry_built_ = true;
}

void CudaViT::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::out);
    if (!f.is_open()) throw std::runtime_error("No se puede escribir: " + path);

    auto params = const_cast<CudaViT*>(this)->parameters();
    int n = (int)params.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(int));

    for (auto* p : params) {
        int ndim = (int)p->shape.size();
        f.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
        f.write(reinterpret_cast<const char*>(p->shape.data()), ndim * sizeof(int));
        std::vector<float> h;
        p->to_host(h);
        f.write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));
    }
    f.flush();
    f.close();
}

void CudaViT::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("No se puede leer: " + path);

    auto params = parameters();
    int n;
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    if (n != (int)params.size())
        throw std::runtime_error("Checkpoint incompatible: distinto numero de parametros");

    for (auto* p : params) {
        int ndim;
        f.read(reinterpret_cast<char*>(&ndim), sizeof(int));
        std::vector<int> sh(ndim);
        f.read(reinterpret_cast<char*>(sh.data()), ndim * sizeof(int));
        int sz = 1; for (int d : sh) sz *= d;
        std::vector<float> h(sz);
        f.read(reinterpret_cast<char*>(h.data()), sz * sizeof(float));
        p->from_host(h);
    }
}

} // namespace vit::cuda
