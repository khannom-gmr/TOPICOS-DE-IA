#include "MLP_CUDA.cuh"
#include "kernels.cuh"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace {

const int THREADS = 256;

inline int blocks1D(int size) { return (size + THREADS - 1) / THREADS; }

// Multiplicación de matrices en GPU
void matMul(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matMulKernel<<<grid, block>>>(A, B, C, M, K, N);
}

void transpose(const float* A, float* B, int rows, int cols) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    transposeKernel<<<grid, block>>>(A, B, rows, cols);
}

} // namespace

MLP_CUDA::MLP_CUDA()
    : W1(784, 128), b1(1, 128), W2(128, 64), b2(1, 64), W3(64, 10), b3(1, 10),
      gW1(784, 128), gb1(1, 128), gW2(128, 64), gb2(1, 64), gW3(64, 10), gb3(1, 10),
      W2t(64, 128), W3t(10, 64),
      z1(1, 128), a1(1, 128), z2(1, 64), a2(1, 64), z3(1, 10), a3(1, 10),
      d1(1, 128), d2(1, 64), d3(1, 10),
      Xt(784, 1), a1t(128, 1), a2t(64, 1),
      currentBatch(0) {
    initWeights();
}

MLP_CUDA::~MLP_CUDA() {}

// Inicializar pesos en CPU (Xavier, seed 42) y copiarlos a la GPU
void MLP_CUDA::initWeights() {
    std::mt19937 gen(42);
    auto xavier = [&](int inSize, int outSize) {
        float limit = std::sqrt(6.0f / (inSize + outSize));
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::vector<float> w(inSize * outSize);
        for (auto& v : w) v = dist(gen);
        return w;
    };

    std::vector<float> w1 = xavier(784, 128);
    std::vector<float> w2 = xavier(128, 64);
    std::vector<float> w3 = xavier(64, 10);

    W1.copyFromHost(w1.data(), (int)w1.size());
    W2.copyFromHost(w2.data(), (int)w2.size());
    W3.copyFromHost(w3.data(), (int)w3.size());
}

void MLP_CUDA::ensureBuffers(int batchSize) {
    if (currentBatch == batchSize) return;
    z1 = CudaMatrix(batchSize, 128);
    a1 = CudaMatrix(batchSize, 128);
    z2 = CudaMatrix(batchSize, 64);
    a2 = CudaMatrix(batchSize, 64);
    z3 = CudaMatrix(batchSize, 10);
    a3 = CudaMatrix(batchSize, 10);
    d1 = CudaMatrix(batchSize, 128);
    d2 = CudaMatrix(batchSize, 64);
    d3 = CudaMatrix(batchSize, 10);
    Xt = CudaMatrix(784, batchSize);
    a1t = CudaMatrix(128, batchSize);
    a2t = CudaMatrix(64, batchSize);
    currentBatch = batchSize;
}

// Propagación hacia adelante
void MLP_CUDA::forwardPass(const float* X, int M) {
    matMul(X, W1.data, z1.data, M, 784, 128);
    addBiasKernel<<<blocks1D(M * 128), THREADS>>>(z1.data, b1.data, M, 128);
    CUDA_CHECK(cudaMemcpy(a1.data, z1.data, sizeof(float) * M * 128, cudaMemcpyDeviceToDevice));
    reluKernel<<<blocks1D(M * 128), THREADS>>>(a1.data, M * 128);

    matMul(a1.data, W2.data, z2.data, M, 128, 64);
    addBiasKernel<<<blocks1D(M * 64), THREADS>>>(z2.data, b2.data, M, 64);
    CUDA_CHECK(cudaMemcpy(a2.data, z2.data, sizeof(float) * M * 64, cudaMemcpyDeviceToDevice));
    reluKernel<<<blocks1D(M * 64), THREADS>>>(a2.data, M * 64);

    matMul(a2.data, W3.data, z3.data, M, 64, 10);
    addBiasKernel<<<blocks1D(M * 10), THREADS>>>(z3.data, b3.data, M, 10);
    CUDA_CHECK(cudaMemcpy(a3.data, z3.data, sizeof(float) * M * 10, cudaMemcpyDeviceToDevice));
    softmaxKernel<<<blocks1D(M), THREADS>>>(a3.data, M, 10);
}

// Propagación hacia atrás y actualización de parámetros
void MLP_CUDA::backpropagation(const float* X, const float* Y, int M, float lr) {
    // Calcular error en la capa de salida
    crossEntropyGradKernel<<<blocks1D(M * 10), THREADS>>>(a3.data, Y, d3.data, M, M * 10);

    transpose(W3.data, W3t.data, 64, 10);
    transpose(a2.data, a2t.data, M, 64);
    matMul(a2t.data, d3.data, gW3.data, 64, M, 10);
    sumColumnsKernel<<<blocks1D(10), THREADS>>>(d3.data, gb3.data, M, 10);

    matMul(d3.data, W3t.data, d2.data, M, 10, 64);
    reluBackwardKernel<<<blocks1D(M * 64), THREADS>>>(d2.data, z2.data, M * 64);

    transpose(W2.data, W2t.data, 128, 64);
    transpose(a1.data, a1t.data, M, 128);
    matMul(a1t.data, d2.data, gW2.data, 128, M, 64);
    sumColumnsKernel<<<blocks1D(64), THREADS>>>(d2.data, gb2.data, M, 64);

    matMul(d2.data, W2t.data, d1.data, M, 64, 128);
    reluBackwardKernel<<<blocks1D(M * 128), THREADS>>>(d1.data, z1.data, M * 128);

    transpose(X, Xt.data, M, 784);
    matMul(Xt.data, d1.data, gW1.data, 784, M, 128);
    sumColumnsKernel<<<blocks1D(128), THREADS>>>(d1.data, gb1.data, M, 128);

    // Actualizar parámetros
    updateWeightsKernel<<<blocks1D(784 * 128), THREADS>>>(W1.data, gW1.data, lr, 784 * 128);
    updateWeightsKernel<<<blocks1D(128), THREADS>>>(b1.data, gb1.data, lr, 128);
    updateWeightsKernel<<<blocks1D(128 * 64), THREADS>>>(W2.data, gW2.data, lr, 128 * 64);
    updateWeightsKernel<<<blocks1D(64), THREADS>>>(b2.data, gb2.data, lr, 64);
    updateWeightsKernel<<<blocks1D(64 * 10), THREADS>>>(W3.data, gW3.data, lr, 64 * 10);
    updateWeightsKernel<<<blocks1D(10), THREADS>>>(b3.data, gb3.data, lr, 10);
}

void MLP_CUDA::train(const std::vector<float>& X, const std::vector<int>& labels,
                     int numSamples, int epochs, float lr, int batchSize) {
    ensureBuffers(batchSize);

    // Subir todos los datos y las etiquetas one-hot a la GPU una sola vez
    CudaMatrix Xdev(numSamples, 784);
    Xdev.copyFromHost(X.data(), numSamples * 784);

    std::vector<float> oneHot(numSamples * 10, 0.0f);
    for (int i = 0; i < numSamples; ++i) oneHot[i * 10 + labels[i]] = 1.0f;
    CudaMatrix Ydev(numSamples, 10);
    Ydev.copyFromHost(oneHot.data(), numSamples * 10);

    std::vector<float> hostA3(batchSize * 10);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epochLoss = 0.0f;
        int batches = 0;

        for (int start = 0; start + batchSize <= numSamples; start += batchSize) {
            const float* Xptr = Xdev.data + start * 784;
            const float* Yptr = Ydev.data + start * 10;

            forwardPass(Xptr, batchSize);

            // Calcular error (cross-entropy) copiando las predicciones al host
            a3.copyToHost(hostA3.data(), batchSize * 10);
            for (int i = 0; i < batchSize; ++i)
                epochLoss -= std::log(hostA3[i * 10 + labels[start + i]] + 1e-12f);
            ++batches;

            backpropagation(Xptr, Yptr, batchSize, lr);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        float acc = computeAccuracy(X, labels, numSamples);
        printf("Epoca %2d | Loss: %.4f | Accuracy: %.2f%%\n",
               epoch, epochLoss / (batches * batchSize), acc);
    }
}

float MLP_CUDA::computeAccuracy(const std::vector<float>& X,
                                const std::vector<int>& labels, int numSamples) {
    if (currentBatch == 0) ensureBuffers(64);
    int B = currentBatch;

    CudaMatrix evalX(B, 784);
    std::vector<float> hostA3(B * 10);
    int correct = 0;

    for (int start = 0; start < numSamples; start += B) {
        int M = std::min(B, numSamples - start);
        evalX.copyFromHost(X.data() + start * 784, M * 784);

        forwardPass(evalX.data, M);
        a3.copyToHost(hostA3.data(), M * 10);

        for (int i = 0; i < M; ++i) {
            int pred = 0;
            float best = hostA3[i * 10];
            for (int j = 1; j < 10; ++j)
                if (hostA3[i * 10 + j] > best) { best = hostA3[i * 10 + j]; pred = j; }
            if (pred == labels[start + i]) ++correct;
        }
    }
    return 100.0f * correct / numSamples;
}
