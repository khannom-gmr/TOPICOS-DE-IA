#ifndef CNN_CUDA_CUH
#define CNN_CUDA_CUH

#include "CudaMatrix.cuh"
#include <vector>

// Arquitectura fija para MNIST:
// Input 1x28x28
//  -> Conv1 (8 filtros 5x5)         -> 8x24x24
//  -> ReLU -> MaxPool 2x2           -> 8x12x12
//  -> Conv2 (16 filtros 5x5)        -> 16x8x8
//  -> ReLU -> MaxPool 2x2           -> 16x4x4
//  -> Flatten                       -> 256
//  -> FC1 256->128 -> ReLU
//  -> FC2 128->10  -> Softmax
class CNN_CUDA {
public:
    CNN_CUDA();
    ~CNN_CUDA();

    void train(const std::vector<float>& X, const std::vector<int>& labels,
               int numSamples, int epochs, float lr, int batchSize);
    float computeAccuracy(const std::vector<float>& X,
                          const std::vector<int>& labels, int numSamples);

private:
    // ---- Dimensiones fijas de la arquitectura ----
    static const int IN_C = 1,  IN_H = 28, IN_W = 28;
    static const int C1_OUT = 16, C1_K = 5, C1_H = 24, C1_W = 24;
    static const int P1_H = 12, P1_W = 12;
    static const int C2_OUT = 32, C2_K = 5, C2_H = 8,  C2_W = 8;
    static const int P2_H = 4,  P2_W = 4;                          // tras pool2
    static const int FLAT = C2_OUT * P2_H * P2_W;                  // 256
    static const int FC1_OUT = 128;
    static const int FC2_OUT = 10;

    static const int C1_SIZE = C1_OUT * C1_H * C1_W;   // 4608
    static const int P1_SIZE = C1_OUT * P1_H * P1_W;   // 1152
    static const int C2_SIZE = C2_OUT * C2_H * C2_W;   // 1024
    static const int P2_SIZE = FLAT;                    // 256

    // ---- Pesos ----
    CudaMatrix convW1, convB1, convW2, convB2;   // pesos de convolucion
    CudaMatrix fcW1, fcb1, fcW2, fcb2;           // pesos fully-connected

    // ---- Gradientes ----
    CudaMatrix gConvW1, gConvB1, gConvW2, gConvB2;
    CudaMatrix gfcW1, gfcb1, gfcW2, gfcb2;

    // ---- Transpuestas auxiliares (para las capas FC, igual que en MLP_CUDA) ----
    CudaMatrix fcW1t, fcW2t, pool2Outt, afc1t;

    // ---- Buffers "batched" (necesarios en el backward, uno por imagen del batch) ----
    CudaMatrix conv1Z;   // (M, C1_SIZE) pre-activacion conv1
    CudaMatrix pool1Out; // (M, P1_SIZE) salida de pool1 (entrada de conv2)
    CudaMatrix conv2Z;   // (M, C2_SIZE) pre-activacion conv2
    CudaMatrix pool2Out; // (M, P2_SIZE) salida de pool2 = entrada de la FC (flatten)
    int* argmax1; // (M, P1_SIZE) indices del maximo de pool1
    int* argmax2; // (M, P2_SIZE) indices del maximo de pool2

    // ---- Buffers batched de la parte FC (igual que MLP_CUDA) ----
    CudaMatrix zfc1, afc1, zfc2, afc2; // afc2 = probabilidades tras softmax
    CudaMatrix dfc2, dfc1;

    // ---- Scratch por-imagen (se reusan en cada iteracion del loop del batch) ----
    CudaMatrix conv1A;  // (1, C1_SIZE) post-relu de conv1, entrada de maxpool1
    CudaMatrix conv2A;  // (1, C2_SIZE) post-relu de conv2, entrada de maxpool2
    CudaMatrix dConv1;  // (1, C1_SIZE) gradiente en conv1 (post -> pre relu)
    CudaMatrix dConv2;  // (1, C2_SIZE) gradiente en conv2 (post -> pre relu)
    CudaMatrix dPool1;  // (1, P1_SIZE) gradiente que llega a pool1Out

    int currentBatch;

    void initWeights();
    void ensureBuffers(int batchSize);
    void freeArgmax();
    void forwardPass(const float* X, int M);
    void backpropagation(const float* X, const float* Y, int M, float lr);
};

#endif
