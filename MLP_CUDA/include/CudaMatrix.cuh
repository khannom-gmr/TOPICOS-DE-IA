#ifndef CUDA_MATRIX_CUH
#define CUDA_MATRIX_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { cudaError_t e = call; if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }

class CudaMatrix {
public:
    CudaMatrix(int rows, int cols);
    ~CudaMatrix();

    // Sin copia, solo move
    CudaMatrix(CudaMatrix&&) noexcept;
    CudaMatrix& operator=(CudaMatrix&&) noexcept;
    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;

    void copyFromHost(const float* hostData, int n);
    void copyToHost(float* hostData, int n) const;
    void fillZeros();

    float* data;
    int rows, cols;
};

#endif
