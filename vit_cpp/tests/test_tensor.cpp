#include "vit/tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                      \
    do {                                                      \
        if (!(cond)) {                                        \
            std::cerr << "[FAIL] " << msg << "\n";           \
            ++g_fail;                                         \
        } else {                                              \
            std::cout << "[PASS] " << msg << "\n";           \
            ++g_pass;                                         \
        }                                                     \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                           \
    CHECK(std::abs((a) - (b)) < (tol), msg)

using namespace vit;

void test_tensor_creation() {
    Tensor t({3, 4});
    CHECK(t.ndim() == 2,       "ndim == 2");
    CHECK(t.numel() == 12,     "numel == 12");
    CHECK(t.shape[0] == 3,     "shape[0] == 3");
    CHECK(t.data[0] == 0.f,    "default zero-init");
}

void test_tensor_fill_access() {
    Tensor t({2, 3}, 5.f);
    CHECK(t.at(0, 0) == 5.f, "fill val at (0,0)");
    t.at(1, 2) = 99.f;
    CHECK(t.at(1, 2) == 99.f, "write/read (1,2)");
}

void test_matmul_identity() {
    Tensor A({2, 2}), I({2, 2});
    A.at(0,0)=1; A.at(0,1)=2; A.at(1,0)=3; A.at(1,1)=4;
    I.at(0,0)=1; I.at(1,1)=1;
    Tensor C = matmul(A, I);
    CHECK_NEAR(C.at(0,0), 1.f, 1e-5f, "matmul identity [0,0]");
    CHECK_NEAR(C.at(1,1), 4.f, 1e-5f, "matmul identity [1,1]");
}

void test_matmul_values() {
    Tensor A({2, 3}), B({3, 2});
    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;
    B.at(0,0)=7; B.at(0,1)=8;
    B.at(1,0)=9; B.at(1,1)=10;
    B.at(2,0)=11; B.at(2,1)=12;
    Tensor C = matmul(A, B);
    CHECK_NEAR(C.at(0,0), 58.f, 1e-4f, "matmul [0,0]");
    CHECK_NEAR(C.at(1,1), 154.f, 1e-4f, "matmul [1,1]");
}

void test_transpose_2d() {
    Tensor A({2, 3});
    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;
    Tensor T = A.transpose_2d();
    CHECK(T.shape[0] == 3 && T.shape[1] == 2, "transpose shape");
    CHECK_NEAR(T.at(0,1), 4.f, 1e-5f, "transpose value");
    CHECK_NEAR(T.at(2,0), 3.f, 1e-5f, "transpose value 2");
}

void test_reshape() {
    Tensor A({6}, 1.f);
    for (int i = 0; i < 6; ++i) A.at(i) = (float)i;
    Tensor B = A.reshape({2, 3});
    CHECK(B.shape[0]==2 && B.shape[1]==3, "reshape dims");
    CHECK_NEAR(B.at(1,2), 5.f, 1e-5f, "reshape value");
}

void test_xavier_init() {
    Tensor W({64, 64});
    W.xavier_uniform_(64, 64);
    float limit = std::sqrt(6.f / 128.f);
    bool all_bounded = true;
    for (float v : W.data)
        if (std::abs(v) > limit + 1e-5f) { all_bounded = false; break; }
    CHECK(all_bounded, "xavier uniform bounded");
    float mean = W.mean();
    CHECK(std::abs(mean) < 0.05f, "xavier mean near 0");
}

int main() {
    std::cout << "=== test_tensor ===\n";
    test_tensor_creation();
    test_tensor_fill_access();
    test_matmul_identity();
    test_matmul_values();
    test_transpose_2d();
    test_reshape();
    test_xavier_init();

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed.\n";
    return g_fail > 0 ? 1 : 0;
}
