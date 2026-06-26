#ifndef MLP_CUDA_CUH
#define MLP_CUDA_CUH

#include "CudaMatrix.cuh"
#include <vector>

class MLP_CUDA {
public:
    MLP_CUDA();   // arquitectura fija 784-128-64-10
    ~MLP_CUDA();

    void train(const std::vector<float>& X, const std::vector<int>& labels,
               int numSamples, int epochs, float lr, int batchSize);
    float computeAccuracy(const std::vector<float>& X,
                          const std::vector<int>& labels, int numSamples);

private:
    // Pesos de cada capa en GPU
    CudaMatrix W1, b1, W2, b2, W3, b3;

    // Gradientes de pesos y bias
    CudaMatrix gW1, gb1, gW2, gb2, gW3, gb3;

    // Transpuestas de los pesos (necesarias en backward)
    CudaMatrix W2t, W3t;

    // Activaciones intermedias y deltas (tamaño = batch × neuronas)
    CudaMatrix z1, a1, z2, a2, z3, a3;
    CudaMatrix d1, d2, d3;
    CudaMatrix Xt, a1t, a2t;

    int currentBatch;

    void initWeights();
    void ensureBuffers(int batchSize);
    void forwardPass(const float* X, int M);
    void backpropagation(const float* X, const float* Y, int M, float lr);
};

#endif
