#include "CudaMatrix.cuh"

CudaMatrix::CudaMatrix(int rows, int cols) : rows(rows), cols(cols) {
    CUDA_CHECK(cudaMalloc(&data, sizeof(float) * rows * cols));
    fillZeros();
}

CudaMatrix::~CudaMatrix() {
    if (data) cudaFree(data);
}

CudaMatrix::CudaMatrix(CudaMatrix&& other) noexcept
    : data(other.data), rows(other.rows), cols(other.cols) {
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

CudaMatrix& CudaMatrix::operator=(CudaMatrix&& other) noexcept {
    if (this != &other) {
        if (data) cudaFree(data);
        data = other.data;
        rows = other.rows;
        cols = other.cols;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

void CudaMatrix::copyFromHost(const float* hostData, int n) {
    CUDA_CHECK(cudaMemcpy(data, hostData, sizeof(float) * n, cudaMemcpyHostToDevice));
}

void CudaMatrix::copyToHost(float* hostData, int n) const {
    CUDA_CHECK(cudaMemcpy(hostData, data, sizeof(float) * n, cudaMemcpyDeviceToHost));
}

void CudaMatrix::fillZeros() {
    CUDA_CHECK(cudaMemset(data, 0, sizeof(float) * rows * cols));
}
