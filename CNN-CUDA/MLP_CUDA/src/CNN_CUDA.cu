#include "CNN_CUDA.cuh"
#include "kernels.cuh"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

namespace {

const int THREADS = 256;
inline int blocks1D(int size) { return (size + THREADS - 1) / THREADS; }

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

// Lanza conv2dForwardKernel para una sola imagen
void conv2dForward(const float* input, int Cin, int H, int W,
                    const float* W_, const float* b, float* output,
                    int Cout, int Ho, int Wo, int K) {
    dim3 block(16, 16);
    dim3 grid((Wo + 15) / 16, (Ho + 15) / 16, Cout);
    conv2dForwardKernel<<<grid, block>>>(input, Cin, H, W, W_, b, output, Cout, Ho, Wo, K);
}

void maxPoolForward(const float* input, int C, int H, int W,
                     float* output, int* argmax, int Ho, int Wo, int pool, int stride) {
    dim3 block(16, 16);
    dim3 grid((Wo + 15) / 16, (Ho + 15) / 16, C);
    maxPool2dForwardKernel<<<grid, block>>>(input, C, H, W, output, argmax, Ho, Wo, pool, stride);
}

void maxPoolBackward(const float* dOutput, const int* argmax, float* dInput,
                      int C, int H, int W, int Ho, int Wo) {
    int total = C * Ho * Wo;
    maxPool2dBackwardKernel<<<blocks1D(total), THREADS>>>(dOutput, argmax, dInput, C, H, W, Ho, Wo);
}

void conv2dBackwardInput(const float* dOutput, int Cout, int Ho, int Wo,
                          const float* W_, int Cin, int K, float* dInput, int H, int W) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16, Cin);
    conv2dBackwardInputKernel<<<grid, block>>>(dOutput, Cout, Ho, Wo, W_, Cin, K, dInput, H, W);
}

void conv2dBackwardWeights(const float* input, int Cin, int H, int W,
                            const float* dOutput, int Cout, int Ho, int Wo, int K,
                            float* dWeights) {
    int total = Cout * Cin * K * K;
    conv2dBackwardWeightsKernel<<<blocks1D(total), THREADS>>>(input, Cin, H, W, dOutput, Cout, Ho, Wo, K, dWeights);
}

void conv2dBackwardBias(const float* dOutput, int Cout, int Ho, int Wo, float* dBias) {
    conv2dBackwardBiasKernel<<<blocks1D(Cout), THREADS>>>(dOutput, Cout, Ho, Wo, dBias);
}

} // namespace

CNN_CUDA::CNN_CUDA()
    : convW1(1, C1_OUT * IN_C * C1_K * C1_K), convB1(1, C1_OUT),
      convW2(1, C2_OUT * C1_OUT * C2_K * C2_K), convB2(1, C2_OUT),
      fcW1(FLAT, FC1_OUT), fcb1(1, FC1_OUT), fcW2(FC1_OUT, FC2_OUT), fcb2(1, FC2_OUT),
      gConvW1(1, C1_OUT * IN_C * C1_K * C1_K), gConvB1(1, C1_OUT),
      gConvW2(1, C2_OUT * C1_OUT * C2_K * C2_K), gConvB2(1, C2_OUT),
      gfcW1(FLAT, FC1_OUT), gfcb1(1, FC1_OUT), gfcW2(FC1_OUT, FC2_OUT), gfcb2(1, FC2_OUT),
      fcW1t(FC1_OUT, FLAT), fcW2t(FC2_OUT, FC1_OUT), pool2Outt(1, 1), afc1t(FC1_OUT, 1),
      conv1Z(1, 1), pool1Out(1, 1), conv2Z(1, 1), pool2Out(1, 1),
      argmax1(nullptr), argmax2(nullptr),
      zfc1(1, FC1_OUT), afc1(1, FC1_OUT), zfc2(1, FC2_OUT), afc2(1, FC2_OUT),
      dfc2(1, FC2_OUT), dfc1(1, FC1_OUT),
      conv1A(1, C1_SIZE), conv2A(1, C2_SIZE),
      dConv1(1, C1_SIZE), dConv2(1, C2_SIZE), dPool1(1, P1_SIZE),
      currentBatch(0) {
    initWeights();
}

CNN_CUDA::~CNN_CUDA() { freeArgmax(); }

void CNN_CUDA::freeArgmax() {
    if (argmax1) { cudaFree(argmax1); argmax1 = nullptr; }
    if (argmax2) { cudaFree(argmax2); argmax2 = nullptr; }
}

// Inicializacion Xavier (CPU) igual que en MLP_CUDA, seed 42 para reproducibilidad
void CNN_CUDA::initWeights() {
    std::mt19937 gen(42);
    auto xavier = [&](int fanIn, int fanOut, int n) {
        float limit = std::sqrt(6.0f / (fanIn + fanOut));
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::vector<float> w(n);
        for (auto& v : w) v = dist(gen);
        return w;
    };

    auto w1 = xavier(IN_C * C1_K * C1_K, C1_OUT, C1_OUT * IN_C * C1_K * C1_K);
    auto w2 = xavier(C1_OUT * C2_K * C2_K, C2_OUT, C2_OUT * C1_OUT * C2_K * C2_K);
    auto wf1 = xavier(FLAT, FC1_OUT, FLAT * FC1_OUT);
    auto wf2 = xavier(FC1_OUT, FC2_OUT, FC1_OUT * FC2_OUT);

    convW1.copyFromHost(w1.data(), (int)w1.size());
    convW2.copyFromHost(w2.data(), (int)w2.size());
    fcW1.copyFromHost(wf1.data(), (int)wf1.size());
    fcW2.copyFromHost(wf2.data(), (int)wf2.size());
    // Biases ya quedan en 0 (CudaMatrix hace fillZeros en el constructor)
}

void CNN_CUDA::ensureBuffers(int batchSize) {
    if (currentBatch == batchSize) return;

    conv1Z   = CudaMatrix(batchSize, C1_SIZE);
    pool1Out = CudaMatrix(batchSize, P1_SIZE);
    conv2Z   = CudaMatrix(batchSize, C2_SIZE);
    pool2Out = CudaMatrix(batchSize, P2_SIZE);

    freeArgmax();
    CUDA_CHECK(cudaMalloc(&argmax1, sizeof(int) * batchSize * P1_SIZE));
    CUDA_CHECK(cudaMalloc(&argmax2, sizeof(int) * batchSize * P2_SIZE));

    zfc1 = CudaMatrix(batchSize, FC1_OUT);
    afc1 = CudaMatrix(batchSize, FC1_OUT);
    zfc2 = CudaMatrix(batchSize, FC2_OUT);
    afc2 = CudaMatrix(batchSize, FC2_OUT);
    dfc2 = CudaMatrix(batchSize, FC2_OUT);
    dfc1 = CudaMatrix(batchSize, FC1_OUT);

    pool2Outt = CudaMatrix(FLAT, batchSize);
    afc1t     = CudaMatrix(FC1_OUT, batchSize);

    currentBatch = batchSize;
}

// Propagacion hacia adelante: conv/pool imagen por imagen, FC en batch
void CNN_CUDA::forwardPass(const float* X, int M) {
    for (int i = 0; i < M; ++i) {
        const float* img = X + i * IN_C * IN_H * IN_W;
        float* c1z = conv1Z.data + i * C1_SIZE;
        float* p1  = pool1Out.data + i * P1_SIZE;
        float* c2z = conv2Z.data + i * C2_SIZE;
        float* p2  = pool2Out.data + i * P2_SIZE;
        int* am1 = argmax1 + i * P1_SIZE;
        int* am2 = argmax2 + i * P2_SIZE;

        // --- Conv1 -> ReLU -> Pool1 ---
        conv2dForward(img, IN_C, IN_H, IN_W, convW1.data, convB1.data, c1z, C1_OUT, C1_H, C1_W, C1_K);
        CUDA_CHECK(cudaMemcpy(conv1A.data, c1z, sizeof(float) * C1_SIZE, cudaMemcpyDeviceToDevice));
        reluKernel<<<blocks1D(C1_SIZE), THREADS>>>(conv1A.data, C1_SIZE);
        maxPoolForward(conv1A.data, C1_OUT, C1_H, C1_W, p1, am1, P1_H, P1_W, 2, 2);

        // --- Conv2 -> ReLU -> Pool2 ---
        conv2dForward(p1, C1_OUT, P1_H, P1_W, convW2.data, convB2.data, c2z, C2_OUT, C2_H, C2_W, C2_K);
        CUDA_CHECK(cudaMemcpy(conv2A.data, c2z, sizeof(float) * C2_SIZE, cudaMemcpyDeviceToDevice));
        reluKernel<<<blocks1D(C2_SIZE), THREADS>>>(conv2A.data, C2_SIZE);
        maxPoolForward(conv2A.data, C2_OUT, C2_H, C2_W, p2, am2, P2_H, P2_W, 2, 2);
    }

    // FC1 -> ReLU -> FC2 -> Softmax (batched, igual que MLP_CUDA) 
    matMul(pool2Out.data, fcW1.data, zfc1.data, M, FLAT, FC1_OUT);
    addBiasKernel<<<blocks1D(M * FC1_OUT), THREADS>>>(zfc1.data, fcb1.data, M, FC1_OUT);
    CUDA_CHECK(cudaMemcpy(afc1.data, zfc1.data, sizeof(float) * M * FC1_OUT, cudaMemcpyDeviceToDevice));
    reluKernel<<<blocks1D(M * FC1_OUT), THREADS>>>(afc1.data, M * FC1_OUT);

    matMul(afc1.data, fcW2.data, zfc2.data, M, FC1_OUT, FC2_OUT);
    addBiasKernel<<<blocks1D(M * FC2_OUT), THREADS>>>(zfc2.data, fcb2.data, M, FC2_OUT);
    CUDA_CHECK(cudaMemcpy(afc2.data, zfc2.data, sizeof(float) * M * FC2_OUT, cudaMemcpyDeviceToDevice));
    softmaxKernel<<<blocks1D(M), THREADS>>>(afc2.data, M, FC2_OUT);
}

// Propagacion hacia atras y actualizacion de parametros
void CNN_CUDA::backpropagation(const float* X, const float* Y, int M, float lr) {
    // ---- Parte FC (identica al patron de MLP_CUDA) ----
    crossEntropyGradKernel<<<blocks1D(M * FC2_OUT), THREADS>>>(afc2.data, Y, dfc2.data, M, M * FC2_OUT);

    transpose(fcW2.data, fcW2t.data, FC1_OUT, FC2_OUT);
    transpose(afc1.data, afc1t.data, M, FC1_OUT);
    matMul(afc1t.data, dfc2.data, gfcW2.data, FC1_OUT, M, FC2_OUT);
    sumColumnsKernel<<<blocks1D(FC2_OUT), THREADS>>>(dfc2.data, gfcb2.data, M, FC2_OUT);

    matMul(dfc2.data, fcW2t.data, dfc1.data, M, FC2_OUT, FC1_OUT);
    reluBackwardKernel<<<blocks1D(M * FC1_OUT), THREADS>>>(dfc1.data, zfc1.data, M * FC1_OUT);

    transpose(fcW1.data, fcW1t.data, FLAT, FC1_OUT);
    transpose(pool2Out.data, pool2Outt.data, M, FLAT);
    matMul(pool2Outt.data, dfc1.data, gfcW1.data, FLAT, M, FC1_OUT);
    sumColumnsKernel<<<blocks1D(FC1_OUT), THREADS>>>(dfc1.data, gfcb1.data, M, FC1_OUT);

    // dPool2Out (M, FLAT): gradiente que entra a la parte convolucional
    CudaMatrix dPool2Out(M, FLAT);
    matMul(dfc1.data, fcW1t.data, dPool2Out.data, M, FC1_OUT, FLAT);

    //  Parte convolucional: se procesa imagen por imagen 
    gConvW1.fillZeros(); gConvB1.fillZeros();
    gConvW2.fillZeros(); gConvB2.fillZeros();

    for (int i = 0; i < M; ++i) {
        const float* img = X + i * IN_C * IN_H * IN_W;
        const float* p1  = pool1Out.data + i * P1_SIZE;
        const float* c1z = conv1Z.data + i * C1_SIZE;
        const float* c2z = conv2Z.data + i * C2_SIZE;
        const int* am1 = argmax1 + i * P1_SIZE;
        const int* am2 = argmax2 + i * P2_SIZE;
        const float* dP2 = dPool2Out.data + i * P2_SIZE;

        //  Backward Pool2 + ReLU(conv2) 
        dConv2.fillZeros();
        maxPoolBackward(dP2, am2, dConv2.data, C2_OUT, C2_H, C2_W, P2_H, P2_W);
        reluBackwardKernel<<<blocks1D(C2_SIZE), THREADS>>>(dConv2.data, c2z, C2_SIZE);

        //  Gradientes de conv2 (pesos, bias) y gradiente hacia pool1Out 
        conv2dBackwardWeights(p1, C1_OUT, P1_H, P1_W, dConv2.data, C2_OUT, C2_H, C2_W, C2_K, gConvW2.data);
        conv2dBackwardBias(dConv2.data, C2_OUT, C2_H, C2_W, gConvB2.data);
        conv2dBackwardInput(dConv2.data, C2_OUT, C2_H, C2_W, convW2.data, C1_OUT, C2_K, dPool1.data, P1_H, P1_W);

        //  Backward Pool1 + ReLU(conv1) 
        dConv1.fillZeros();
        maxPoolBackward(dPool1.data, am1, dConv1.data, C1_OUT, C1_H, C1_W, P1_H, P1_W);
        reluBackwardKernel<<<blocks1D(C1_SIZE), THREADS>>>(dConv1.data, c1z, C1_SIZE);

        // --- Gradientes de conv1 (no hace falta dX: es la primera capa) ---
        conv2dBackwardWeights(img, IN_C, IN_H, IN_W, dConv1.data, C1_OUT, C1_H, C1_W, C1_K, gConvW1.data);
        conv2dBackwardBias(dConv1.data, C1_OUT, C1_H, C1_W, gConvB1.data);
    }

    //  Actualizar todos los parametros 
    updateWeightsKernel<<<blocks1D(convW1.rows * convW1.cols), THREADS>>>(convW1.data, gConvW1.data, lr, convW1.rows * convW1.cols);
    updateWeightsKernel<<<blocks1D(convB1.cols), THREADS>>>(convB1.data, gConvB1.data, lr, convB1.cols);
    updateWeightsKernel<<<blocks1D(convW2.rows * convW2.cols), THREADS>>>(convW2.data, gConvW2.data, lr, convW2.rows * convW2.cols);
    updateWeightsKernel<<<blocks1D(convB2.cols), THREADS>>>(convB2.data, gConvB2.data, lr, convB2.cols);

    updateWeightsKernel<<<blocks1D(FLAT * FC1_OUT), THREADS>>>(fcW1.data, gfcW1.data, lr, FLAT * FC1_OUT);
    updateWeightsKernel<<<blocks1D(FC1_OUT), THREADS>>>(fcb1.data, gfcb1.data, lr, FC1_OUT);
    updateWeightsKernel<<<blocks1D(FC1_OUT * FC2_OUT), THREADS>>>(fcW2.data, gfcW2.data, lr, FC1_OUT * FC2_OUT);
    updateWeightsKernel<<<blocks1D(FC2_OUT), THREADS>>>(fcb2.data, gfcb2.data, lr, FC2_OUT);
}

void CNN_CUDA::train(const std::vector<float>& X, const std::vector<int>& labels,
                     int numSamples, int epochs, float lr, int batchSize,
                     const std::string& csvPath) {
    ensureBuffers(batchSize);

    CudaMatrix Xdev(numSamples, IN_C * IN_H * IN_W);
    Xdev.copyFromHost(X.data(), numSamples * IN_C * IN_H * IN_W);

    std::vector<float> oneHot(numSamples * FC2_OUT, 0.0f);
    for (int i = 0; i < numSamples; ++i) oneHot[i * FC2_OUT + labels[i]] = 1.0f;
    CudaMatrix Ydev(numSamples, FC2_OUT);
    Ydev.copyFromHost(oneHot.data(), numSamples * FC2_OUT);

    std::vector<float> hostA(batchSize * FC2_OUT);
    std::vector<int> historyEpoch;
    std::vector<float> historyLoss, historyAcc;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epochLoss = 0.0f;
        int batches = 0;

        for (int start = 0; start + batchSize <= numSamples; start += batchSize) {
            const float* Xptr = Xdev.data + start * IN_C * IN_H * IN_W;
            const float* Yptr = Ydev.data + start * FC2_OUT;

            forwardPass(Xptr, batchSize);

            afc2.copyToHost(hostA.data(), batchSize * FC2_OUT);
            for (int i = 0; i < batchSize; ++i)
                epochLoss -= std::log(hostA[i * FC2_OUT + labels[start + i]] + 1e-12f);
            ++batches;

            backpropagation(Xptr, Yptr, batchSize, lr);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        float avgLoss = epochLoss / (batches * batchSize);
        float acc = computeAccuracy(X, labels, numSamples);
        printf("Epoca %2d | Loss: %.4f | Accuracy: %.2f%%\n", epoch, avgLoss, acc);

        historyEpoch.push_back(epoch);
        historyLoss.push_back(avgLoss);
        historyAcc.push_back(acc);
    }

    if (!csvPath.empty()) {
        std::ofstream csv(csvPath);
        if (csv.is_open()) {
            csv << "epoch,loss,accuracy\n";
            for (size_t i = 0; i < historyEpoch.size(); ++i)
                csv << historyEpoch[i] << "," << historyLoss[i] << "," << historyAcc[i] << "\n";
            csv.close();
            printf("Curva de aprendizaje guardada en %s\n", csvPath.c_str());
        } else {
            printf("Aviso: no se pudo escribir %s\n", csvPath.c_str());
        }
    }
}

float CNN_CUDA::computeAccuracy(const std::vector<float>& X,
                                const std::vector<int>& labels, int numSamples) {
    if (currentBatch == 0) ensureBuffers(64);
    int B = currentBatch;

    CudaMatrix evalX(B, IN_C * IN_H * IN_W);
    std::vector<float> hostA(B * FC2_OUT);
    int correct = 0;

    for (int start = 0; start < numSamples; start += B) {
        int M = std::min(B, numSamples - start);
        evalX.copyFromHost(X.data() + start * IN_C * IN_H * IN_W, M * IN_C * IN_H * IN_W);

        forwardPass(evalX.data, M);
        afc2.copyToHost(hostA.data(), M * FC2_OUT);

        for (int i = 0; i < M; ++i) {
            int pred = 0;
            float best = hostA[i * FC2_OUT];
            for (int j = 1; j < FC2_OUT; ++j)
                if (hostA[i * FC2_OUT + j] > best) { best = hostA[i * FC2_OUT + j]; pred = j; }
            if (pred == labels[start + i]) ++correct;
        }
    }
    return 100.0f * correct / numSamples;
}

float CNN_CUDA::evaluate(const std::vector<float>& X, const std::vector<int>& labels,
                         int numSamples) {
    if (currentBatch == 0) ensureBuffers(64);
    int B = currentBatch;

    CudaMatrix evalX(B, IN_C * IN_H * IN_W);
    std::vector<float> hostA(B * FC2_OUT);

    std::vector<std::vector<int>> confusion(FC2_OUT, std::vector<int>(FC2_OUT, 0));
    double totalLoss = 0.0;
    int correct = 0;

    for (int start = 0; start < numSamples; start += B) {
        int M = std::min(B, numSamples - start);
        evalX.copyFromHost(X.data() + start * IN_C * IN_H * IN_W, M * IN_C * IN_H * IN_W);

        forwardPass(evalX.data, M);
        afc2.copyToHost(hostA.data(), M * FC2_OUT);

        for (int i = 0; i < M; ++i) {
            int trueLabel = labels[start + i];
            int pred = 0;
            float best = hostA[i * FC2_OUT];
            for (int j = 1; j < FC2_OUT; ++j)
                if (hostA[i * FC2_OUT + j] > best) { best = hostA[i * FC2_OUT + j]; pred = j; }

            confusion[trueLabel][pred]++;
            if (pred == trueLabel) ++correct;
            totalLoss -= std::log(hostA[i * FC2_OUT + trueLabel] + 1e-12);
        }
    }

    float acc = (float)correct / numSamples;
    float avgLoss = (float)(totalLoss / numSamples);

    // ---- Precision / Recall / F1 por clase ----
    std::vector<int> support(FC2_OUT, 0);
    std::vector<double> precision(FC2_OUT, 0.0), recall(FC2_OUT, 0.0), f1(FC2_OUT, 0.0);

    for (int c = 0; c < FC2_OUT; ++c) {
        int rowSum = 0; // soporte real de la clase c
        for (int j = 0; j < FC2_OUT; ++j) rowSum += confusion[c][j];
        support[c] = rowSum;

        int colSum = 0; // veces que se predijo la clase c
        for (int r = 0; r < FC2_OUT; ++r) colSum += confusion[r][c];

        double tp = confusion[c][c];
        precision[c] = colSum > 0 ? tp / colSum : 0.0;
        recall[c]    = rowSum > 0 ? tp / rowSum : 0.0;
        f1[c]        = (precision[c] + recall[c] > 0)
                       ? 2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
                       : 0.0;
    }

    double macroP = 0, macroR = 0, macroF1 = 0;
    double weightedP = 0, weightedR = 0, weightedF1 = 0;
    for (int c = 0; c < FC2_OUT; ++c) {
        macroP += precision[c]; macroR += recall[c]; macroF1 += f1[c];
        weightedP += precision[c] * support[c];
        weightedR += recall[c] * support[c];
        weightedF1 += f1[c] * support[c];
    }
    macroP /= FC2_OUT; macroR /= FC2_OUT; macroF1 /= FC2_OUT;
    weightedP /= numSamples; weightedR /= numSamples; weightedF1 /= numSamples;

    // ---- Impresion del reporte ----
    printf("\n=== REPORTE DE EVALUACION ===\n");
    printf("Exactitud (accuracy): %.4f  (%d/%d)\n", acc, correct, numSamples);
    printf("Clase   Precision    Recall        F1   Soporte\n");
    printf("-------------------------------------------------\n");
    for (int c = 0; c < FC2_OUT; ++c) {
        printf("%5d %10.4f %10.4f %9.4f %9d\n", c, precision[c], recall[c], f1[c], support[c]);
    }
    printf("-------------------------------------------------\n");
    printf("macro %10.4f %10.4f %9.4f %9d\n", macroP, macroR, macroF1, numSamples);
    printf("ponder. %8.4f %10.4f %9.4f %9d\n", weightedP, weightedR, weightedF1, numSamples);

    printf("Matriz de confusion (filas = real, columnas = predicho):\n");
    printf("     ");
    for (int c = 0; c < FC2_OUT; ++c) printf("%5d", c);
    printf("\n");
    for (int r = 0; r < FC2_OUT; ++r) {
        printf("%4d ", r);
        for (int c = 0; c < FC2_OUT; ++c) printf("%5d", confusion[r][c]);
        printf("\n");
    }
    printf("=======================================================\n");
    printf("Perdida final en prueba: %.4f\n", avgLoss);

    return avgLoss;
}
