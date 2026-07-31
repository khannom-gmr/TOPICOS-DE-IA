#pragma once
#include "vit/tensor.hpp"
#include "vit/layers.hpp"
#include "vit/attention.hpp"
#include <vector>

namespace vit {

// -----------------------------------------------------------------------
// Configuracion de Vision Transformer (ViT) - Slide 101-110
// -----------------------------------------------------------------------
struct ViTConfig {
    int image_size   = 28;   // MNIST 28x28
    int patch_size   = 7;    // Parches de 7x7 (16 parches en total)
    int num_channels = 1;    // Escala de grises
    int embed_dim    = 64;
    int num_heads    = 4;
    int num_layers   = 4;
    int mlp_dim      = 128;
    int num_classes  = 10;

    int num_patches() const {
        int n = image_size / patch_size;
        return n * n; // 4*4 = 16 parches
    }
    int seq_len()   const { return num_patches() + 1; }  // +1 para token CLS
    int head_dim()  const { return embed_dim / num_heads; }
    int patch_dim() const { return patch_size * patch_size * num_channels; }
};

// Patch Embedding: Aplana parches e implementa proyeccion lineal (Slide 103)
class PatchEmbedding {
public:
    PatchEmbedding() = default;
    explicit PatchEmbedding(const ViTConfig& cfg);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
    void   zero_grad();

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

private:
    ViTConfig cfg_;
    Linear    proj_;
    Tensor    last_patches_;
};

// Bloque Transformer Pre-Norm (Slides 112-113):
//   residual = x + MHSA(LN1(x))
//   out      = residual + MLP(LN2(residual))
class TransformerBlock {
public:
    TransformerBlock() = default;
    explicit TransformerBlock(const ViTConfig& cfg);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& g);
    void   zero_grad();

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

private:
    LayerNorm          ln1_, ln2_;
    MultiHeadAttention mhsa_;
    Linear             ff1_, ff2_;

    Tensor last_x_;
    Tensor last_ln1_out_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_ln2_out_;
    Tensor last_ff1_out_;
    Tensor last_ff2_in_;
};

// Vision Transformer Completo (Slides 101-110)
class ViT {
public:
    ViT() = default;
    explicit ViT(const ViTConfig& cfg);

    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad_logits);

    void zero_grad();
    void set_training(bool training) { training_ = training; }

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

    const ViTConfig& config() const { return cfg_; }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    ViTConfig cfg_;
    bool      training_ = true;

    PatchEmbedding                patch_embed_;
    Tensor                        cls_token_;
    Tensor                        pos_embed_;
    Tensor                        d_cls_token_;
    Tensor                        d_pos_embed_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm                     final_norm_;
    Linear                        classifier_;

    Tensor last_patches_;
    Tensor last_tokens_;
    Tensor last_cls_out_;
};

} // namespace vit
