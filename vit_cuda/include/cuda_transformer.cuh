#pragma once
#include "cuda_tensor.cuh"
#include "cuda_layers.cuh"
#include "cuda_attention.cuh"
#include "vit/transformer.hpp"
#include <cublas_v2.h>
#include <vector>
#include <string>

namespace vit::cuda {

class CudaTransformerBlock {
public:
    CudaTransformerBlock() = default;
    explicit CudaTransformerBlock(const vit::ViTConfig& cfg);

    CudaTensor forward(cublasHandle_t handle, const CudaTensor& x,
                       int B, int S);
    CudaTensor backward(cublasHandle_t handle, const CudaTensor& g,
                        int B, int S);

    void zero_grad();
    void get_params(std::vector<CudaTensor*>& params,
                    std::vector<CudaTensor*>& grads);

private:
    vit::ViTConfig      cfg_;
    CudaLayerNorm       ln1_, ln2_;
    CudaMultiHeadAttention mhsa_;
    CudaLinear          ff1_, ff2_;

    CudaTensor last_x_;
    CudaTensor last_ln1_out_;
    CudaTensor last_attn_out_;
    CudaTensor last_res1_;
    CudaTensor last_ln2_out_;
    CudaTensor last_ff1_pre_gelu_;
    CudaTensor last_ff1_out_;
};

class CudaViT {
public:
    CudaViT() = default;
    explicit CudaViT(const vit::ViTConfig& cfg);

    ~CudaViT();

    CudaTensor forward(const std::vector<float>& x_host, int B);

    void backward(const CudaTensor& grad_logits, int B);

    void zero_grad();

    std::pair<float, int> loss_and_accuracy(
        const CudaTensor& logits,
        const std::vector<int>& labels, int B);

    std::vector<CudaTensor*> parameters();
    std::vector<CudaTensor*> gradients();

    const vit::ViTConfig& config() const { return cfg_; }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    vit::ViTConfig    cfg_;
    cublasHandle_t    cublas_;

    CudaLinear        patch_proj_;
    CudaTensor        cls_token_;
    CudaTensor        pos_embed_;
    CudaTensor        d_cls_token_;
    CudaTensor        d_pos_embed_;

    std::vector<CudaTransformerBlock> blocks_;
    CudaLayerNorm     final_norm_;
    CudaLinear        classifier_;

    std::vector<CudaTensor*> all_params_;
    std::vector<CudaTensor*> all_grads_;
    bool                     registry_built_ = false;

    CudaTensor last_x_device_;
    CudaTensor last_tokens_;
    CudaTensor last_blocks_out_;
    CudaTensor last_cls_;
    int        last_B_ = 0;

    void build_registry();
    void make_cublas();

    static CudaTensor extract_patches(const std::vector<float>& x_host,
                                      int B, const vit::ViTConfig& cfg);
};

} // namespace vit::cuda
