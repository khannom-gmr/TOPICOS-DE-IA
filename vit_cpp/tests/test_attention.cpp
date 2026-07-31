#include "vit/layers.hpp"
#include "vit/attention.hpp"
#include "vit/transformer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; ++g_fail; } \
         else { std::cout << "[PASS] " << msg << "\n"; ++g_pass; } } while(0)
#define CHECK_NEAR(a, b, tol, msg) CHECK(std::abs((a)-(b)) < (tol), msg)

using namespace vit;

void test_softmax_sums_to_one() {
    Tensor x({2, 4});
    x.at(0,0)=1; x.at(0,1)=2; x.at(0,2)=3; x.at(0,3)=4;
    x.at(1,0)=-1; x.at(1,1)=0; x.at(1,2)=1; x.at(1,3)=2;
    Tensor s = softmax(x);
    float sum0 = s.at(0,0)+s.at(0,1)+s.at(0,2)+s.at(0,3);
    float sum1 = s.at(1,0)+s.at(1,1)+s.at(1,2)+s.at(1,3);
    CHECK_NEAR(sum0, 1.f, 1e-5f, "softmax row 0 sums to 1");
    CHECK_NEAR(sum1, 1.f, 1e-5f, "softmax row 1 sums to 1");
}

void test_cross_entropy_correct_class() {
    Tensor logits({2, 3});
    logits.at(0,0)=10; logits.at(0,1)=-10; logits.at(0,2)=-10;
    logits.at(1,0)=-10; logits.at(1,1)=-10; logits.at(1,2)=10;
    std::vector<int> labels = {0, 2};
    auto [loss, grad] = cross_entropy(logits, labels);
    CHECK(loss < 0.01f, "CE loss near 0 for perfect predictions");
    CHECK(grad.at(0, 0) <= 1e-4f, "grad at correct class is <= 0 (row 0)");
    CHECK(grad.at(1, 2) <= 1e-4f, "grad at correct class is <= 0 (row 1)");
}

void test_linear_forward_shape() {
    Linear L(8, 4);
    Tensor x({3, 8}); x.randn_(0.1f);
    Tensor y = L.forward(x);
    CHECK(y.shape[0]==3 && y.shape[1]==4, "linear output shape [3,4]");
}

void test_layernorm_mean_var() {
    LayerNorm ln(16);
    Tensor x({2, 3, 16}); x.randn_(1.f);
    Tensor y = ln.forward(x);
    for (int b = 0; b < 2; ++b) {
        for (int n = 0; n < 3; ++n) {
            float mu = 0.f;
            for (int d = 0; d < 16; ++d) mu += y.at(b,n,d);
            mu /= 16;
            CHECK_NEAR(mu, 0.f, 1e-4f, "layernorm mean ~ 0");
        }
    }
}

void test_mhsa_output_shape() {
    MultiHeadAttention mhsa(8, 2);
    Tensor x({2, 5, 8}); x.randn_(0.1f);
    Tensor y = mhsa.forward(x);
    CHECK(y.shape[0]==2 && y.shape[1]==5 && y.shape[2]==8,
          "MHSA output shape [2,5,8]");
}

void test_vit_forward_shape() {
    ViTConfig cfg;
    ViT model(cfg);
    Tensor x({4, 784}); x.randn_(0.1f);
    Tensor logits = model.forward(x);
    CHECK(logits.shape[0]==4 && logits.shape[1]==10,
          "ViT output shape [4,10]");
}

void test_vit_backward_no_nan() {
    ViTConfig cfg;
    ViT model(cfg);
    Tensor x({2, 784}); x.randn_(0.1f);
    std::vector<int> labels = {3, 7};

    model.zero_grad();
    Tensor logits = model.forward(x);
    auto [loss, grad] = cross_entropy(logits, labels);
    model.backward(grad);

    bool no_nan = true;
    for (auto* g : model.gradients())
        for (float v : g->data)
            if (std::isnan(v) || std::isinf(v)) { no_nan = false; break; }

    CHECK(no_nan, "no NaN/Inf in gradients after backward");
    CHECK(loss > 0.f && loss < 5.f, "loss in reasonable range [0, 5]");
}

int main() {
    std::cout << "=== test_attention ===\n";
    test_softmax_sums_to_one();
    test_cross_entropy_correct_class();
    test_linear_forward_shape();
    test_layernorm_mean_var();
    test_mhsa_output_shape();
    test_vit_forward_shape();
    test_vit_backward_no_nan();

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed.\n";
    return g_fail > 0 ? 1 : 0;
}
