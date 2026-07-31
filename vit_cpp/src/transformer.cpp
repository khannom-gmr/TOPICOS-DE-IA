#include "vit/transformer.hpp"
#include <cassert>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vit {

PatchEmbedding::PatchEmbedding(const ViTConfig& cfg)
    : cfg_(cfg), proj_(cfg.patch_dim(), cfg.embed_dim) {}

Tensor PatchEmbedding::forward(const Tensor& x) {
    int B  = x.shape[0];
    int N  = cfg_.num_patches();
    int PD = cfg_.patch_dim();
    int D  = cfg_.embed_dim;

    int img_h = cfg_.image_size, img_w = cfg_.image_size;
    int ph = cfg_.patch_size,    pw = cfg_.patch_size;
    int nh = img_h / ph, nw = img_w / pw;

    Tensor patches({B * N, PD});
    for (int b = 0; b < B; ++b) {
        for (int py = 0; py < nh; ++py) {
            for (int px = 0; px < nw; ++px) {
                int patch_idx = py * nw + px;
                for (int dy = 0; dy < ph; ++dy) {
                    for (int dx = 0; dx < pw; ++dx) {
                        int pixel_idx = (py * ph + dy) * img_w + (px * pw + dx);
                        int patch_dim_idx = dy * pw + dx;
                        patches.at(b * N + patch_idx, patch_dim_idx)
                            = x.at(b, pixel_idx);
                    }
                }
            }
        }
    }
    last_patches_ = patches;

    Tensor emb = proj_.forward(patches);
    return emb.reshape({B, N, D});
}

Tensor PatchEmbedding::backward(const Tensor& grad_out) {
    int B  = grad_out.shape[0];
    int N  = cfg_.num_patches();

    Tensor grad_flat = grad_out.reshape({B * N, cfg_.embed_dim});
    Tensor d_patches = proj_.backward(grad_flat);

    int img_h = cfg_.image_size, img_w = cfg_.image_size;
    int ph = cfg_.patch_size,    pw = cfg_.patch_size;
    int nh = img_h / ph, nw = img_w / pw;

    Tensor d_x({B, img_h * img_w});
    for (int b = 0; b < B; ++b) {
        for (int py = 0; py < nh; ++py) {
            for (int px = 0; px < nw; ++px) {
                int patch_idx = py * nw + px;
                for (int dy = 0; dy < ph; ++dy) {
                    for (int dx = 0; dx < pw; ++dx) {
                        int pixel_idx     = (py * ph + dy) * img_w + (px * pw + dx);
                        int patch_dim_idx = dy * pw + dx;
                        d_x.at(b, pixel_idx) +=
                            d_patches.at(b * N + patch_idx, patch_dim_idx);
                    }
                }
            }
        }
    }
    return d_x;
}

void PatchEmbedding::zero_grad() { proj_.zero_grad(); }

std::vector<Tensor*> PatchEmbedding::parameters() {
    return {&proj_.W, &proj_.b};
}

std::vector<Tensor*> PatchEmbedding::gradients() {
    return {&proj_.dW, &proj_.db};
}

TransformerBlock::TransformerBlock(const ViTConfig& cfg)
    : ln1_(cfg.embed_dim),
      ln2_(cfg.embed_dim),
      mhsa_(cfg.embed_dim, cfg.num_heads),
      ff1_(cfg.embed_dim, cfg.mlp_dim),
      ff2_(cfg.mlp_dim, cfg.embed_dim) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    last_x_ = x;

    last_ln1_out_  = ln1_.forward(x);
    last_attn_out_ = mhsa_.forward(last_ln1_out_);
    last_res1_     = add(x, last_attn_out_);

    last_ln2_out_ = ln2_.forward(last_res1_);
    int B = x.shape[0], N = x.shape[1], D = x.shape[2];
    Tensor ln2_flat = last_ln2_out_.reshape({B * N, D});

    last_ff2_in_ = ff1_.forward(ln2_flat);
    last_ff1_out_ = gelu(last_ff2_in_);
    Tensor ff2_out = ff2_.forward(last_ff1_out_);

    Tensor mlp_out = ff2_out.reshape({B, N, D});
    return add(last_res1_, mlp_out);
}

Tensor TransformerBlock::backward(const Tensor& g) {
    int B = last_x_.shape[0], N = last_x_.shape[1], D = last_x_.shape[2];

    Tensor g_mlp    = g.reshape({B * N, D});
    Tensor d_ff2    = ff2_.backward(g_mlp);
    Tensor d_ff1    = gelu_backward(last_ff2_in_, d_ff2);
    Tensor d_ln2_in = ff1_.backward(d_ff1);
    d_ln2_in = d_ln2_in.reshape({B, N, D});

    Tensor d_res1   = ln2_.backward(d_ln2_in);
    d_res1.add_inplace_(g);

    Tensor d_attn  = mhsa_.backward(d_res1);
    Tensor d_ln1   = ln1_.backward(d_attn);
    d_ln1.add_inplace_(d_res1);

    return d_ln1;
}

void TransformerBlock::zero_grad() {
    ln1_.zero_grad(); ln2_.zero_grad();
    mhsa_.zero_grad();
    ff1_.zero_grad(); ff2_.zero_grad();
}

std::vector<Tensor*> TransformerBlock::parameters() {
    auto p = mhsa_.parameters();
    p.insert(p.end(), {&ln1_.gamma, &ln1_.beta, &ln2_.gamma, &ln2_.beta,
                        &ff1_.W, &ff1_.b, &ff2_.W, &ff2_.b});
    return p;
}

std::vector<Tensor*> TransformerBlock::gradients() {
    auto g = mhsa_.gradients();
    g.insert(g.end(), {&ln1_.dgamma, &ln1_.dbeta, &ln2_.dgamma, &ln2_.dbeta,
                        &ff1_.dW, &ff1_.db, &ff2_.dW, &ff2_.db});
    return g;
}

ViT::ViT(const ViTConfig& cfg) : cfg_(cfg) {
    patch_embed_ = PatchEmbedding(cfg);

    cls_token_  = Tensor({1, cfg.embed_dim});
    pos_embed_  = Tensor({cfg.seq_len(), cfg.embed_dim});
    d_cls_token_ = Tensor({1, cfg.embed_dim}, 0.f);
    d_pos_embed_ = Tensor({cfg.seq_len(), cfg.embed_dim}, 0.f);
    cls_token_.randn_(0.02f);
    pos_embed_.randn_(0.02f);

    blocks_.reserve(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; ++i)
        blocks_.emplace_back(cfg);

    final_norm_ = LayerNorm(cfg.embed_dim);
    classifier_ = Linear(cfg.embed_dim, cfg.num_classes);
}

Tensor ViT::forward(const Tensor& x) {
    int B = x.shape[0];
    int N = cfg_.num_patches();
    int D = cfg_.embed_dim;
    int S = cfg_.seq_len();

    last_patches_ = patch_embed_.forward(x);

    Tensor tokens({B, S, D});
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d)
            tokens.at(b, 0, d) = cls_token_.at(0, d);
        for (int n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
                tokens.at(b, n + 1, d) = last_patches_.at(b, n, d);
    }

    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < D; ++d)
                tokens.at(b, s, d) += pos_embed_.at(s, d);

    last_tokens_ = tokens;

    Tensor h = tokens;
    for (auto& block : blocks_)
        h = block.forward(h);

    Tensor normed = final_norm_.forward(h);
    Tensor cls({B, D});
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
            cls.at(b, d) = normed.at(b, 0, d);
    last_cls_out_ = cls;

    return classifier_.forward(cls);
}

void ViT::backward(const Tensor& grad_logits) {
    int B = grad_logits.shape[0];
    int D = cfg_.embed_dim;
    int N = cfg_.num_patches();
    int S = cfg_.seq_len();

    Tensor d_cls = classifier_.backward(grad_logits);

    Tensor d_normed({B, S, D}, 0.f);
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
            d_normed.at(b, 0, d) = d_cls.at(b, d);

    Tensor d_h = final_norm_.backward(d_normed);

    Tensor g = d_h;
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i)
        g = blocks_[i].backward(g);

    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int d = 0; d < D; ++d)
                d_pos_embed_.at(s, d) += g.at(b, s, d);

    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d)
            d_cls_token_.at(0, d) += g.at(b, 0, d);

    Tensor d_patches({B, N, D});
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
                d_patches.at(b, n, d) = g.at(b, n + 1, d);

    patch_embed_.backward(d_patches);
}

void ViT::zero_grad() {
    patch_embed_.zero_grad();
    d_cls_token_.zero_();
    d_pos_embed_.zero_();
    for (auto& block : blocks_) block.zero_grad();
    final_norm_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> ViT::parameters() {
    std::vector<Tensor*> p = patch_embed_.parameters();
    p.push_back(&cls_token_);
    p.push_back(&pos_embed_);
    for (auto& block : blocks_) {
        auto bp = block.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto fp = final_norm_.parameters();
    p.insert(p.end(), fp.begin(), fp.end());
    p.push_back(&classifier_.W);
    p.push_back(&classifier_.b);
    return p;
}

std::vector<Tensor*> ViT::gradients() {
    std::vector<Tensor*> g = patch_embed_.gradients();
    g.push_back(&d_cls_token_);
    g.push_back(&d_pos_embed_);
    for (auto& block : blocks_) {
        auto bg = block.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto fg = final_norm_.gradients();
    g.insert(g.end(), fg.begin(), fg.end());
    g.push_back(&classifier_.dW);
    g.push_back(&classifier_.db);
    return g;
}

void ViT::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write checkpoint: " + path);

    auto write_tensor = [&](const Tensor& t) {
        int ndim = t.ndim();
        f.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
        f.write(reinterpret_cast<const char*>(t.shape.data()), ndim * sizeof(int));
        f.write(reinterpret_cast<const char*>(t.data.data()),
                t.numel() * sizeof(float));
    };

    auto all_params = const_cast<ViT*>(this)->parameters();
    int n = static_cast<int>(all_params.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(int));
    for (auto* p : all_params) write_tensor(*p);
}

void ViT::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read checkpoint: " + path);

    auto read_tensor = [&](Tensor& t) {
        int ndim;
        f.read(reinterpret_cast<char*>(&ndim), sizeof(int));
        t.shape.resize(ndim);
        f.read(reinterpret_cast<char*>(t.shape.data()), ndim * sizeof(int));
        int sz = 1;
        for (int d : t.shape) sz *= d;
        t.data.resize(sz);
        f.read(reinterpret_cast<char*>(t.data.data()), sz * sizeof(float));
    };

    auto all_params = parameters();
    int n;
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    if (n != static_cast<int>(all_params.size()))
        throw std::runtime_error("Checkpoint parameter count mismatch");
    for (auto* p : all_params) read_tensor(*p);
}

} // namespace vit
