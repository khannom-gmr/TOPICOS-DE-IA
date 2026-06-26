#include "MLP_CPU.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace {

float reluFn(float x) { return x > 0.0f ? x : 0.0f; }
float reluDerivFn(float x) { return x > 0.0f ? 1.0f : 0.0f; }

// Sumar bias a cada fila
void addBias(Matrix& m, const Matrix& bias) {
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            m(i, j) += bias(0, j);
}

// Función de activación: softmax por filas con estabilidad numérica
Matrix softmax(const Matrix& x) {
    Matrix out(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        float maxVal = x(i, 0);
        for (int j = 1; j < x.cols(); ++j) maxVal = std::max(maxVal, x(i, j));
        float sum = 0.0f;
        for (int j = 0; j < x.cols(); ++j) {
            float e = std::exp(x(i, j) - maxVal);
            out(i, j) = e;
            sum += e;
        }
        for (int j = 0; j < x.cols(); ++j) out(i, j) /= sum;
    }
    return out;
}

// Inicialización Xavier uniforme
Matrix xavier(int inSize, int outSize, std::mt19937& gen) {
    float limit = std::sqrt(6.0f / (inSize + outSize));
    std::uniform_real_distribution<float> dist(-limit, limit);
    Matrix m(inSize, outSize);
    for (int i = 0; i < inSize; ++i)
        for (int j = 0; j < outSize; ++j)
            m(i, j) = dist(gen);
    return m;
}

Matrix colSum(const Matrix& m) {
    Matrix s(1, m.cols());
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            s(0, j) += m(i, j);
    return s;
}

} // namespace

MLP_CPU::MLP_CPU() { initWeights(); }

// Inicializar pesos
void MLP_CPU::initWeights() {
    std::mt19937 gen(42);
    W1 = xavier(784, 128, gen);
    b1 = Matrix::zeros(1, 128);
    W2 = xavier(128, 64, gen);
    b2 = Matrix::zeros(1, 64);
    W3 = xavier(64, 10, gen);
    b3 = Matrix::zeros(1, 10);
}

// Propagación hacia adelante
Matrix MLP_CPU::forwardPass(const Matrix& input, Matrix& z1, Matrix& a1,
                            Matrix& z2, Matrix& a2, Matrix& z3) const {
    z1 = input.multiply(W1);
    addBias(z1, b1);
    a1 = z1.apply(reluFn);

    z2 = a1.multiply(W2);
    addBias(z2, b2);
    a2 = z2.apply(reluFn);

    z3 = a2.multiply(W3);
    addBias(z3, b3);
    return softmax(z3);
}

void MLP_CPU::train(const std::vector<float>& X, const std::vector<int>& labels,
                    int numSamples, int epochs, float lr, int batchSize) {
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epochLoss = 0.0f;
        int batches = 0;

        for (int start = 0; start + batchSize <= numSamples; start += batchSize) {
            int M = batchSize;

            // Construir el minibatch y las etiquetas one-hot
            Matrix input(M, 784), Y(M, 10);
            for (int i = 0; i < M; ++i) {
                int s = start + i;
                for (int j = 0; j < 784; ++j) input(i, j) = X[s * 784 + j];
                Y(i, labels[s]) = 1.0f;
            }

            Matrix z1, a1, z2, a2, z3;
            Matrix a3 = forwardPass(input, z1, a1, z2, a2, z3);

            // Calcular error (cross-entropy)
            for (int i = 0; i < M; ++i)
                epochLoss -= std::log(a3(i, labels[start + i]) + 1e-12f);
            ++batches;

            // Propagación hacia atrás
            Matrix delta3 = (a3 - Y) * (1.0f / M);
            Matrix gradW3 = a2.transpose().multiply(delta3);
            Matrix gradb3 = colSum(delta3);

            Matrix delta2 = delta3.multiply(W3.transpose());
            for (int i = 0; i < delta2.rows(); ++i)
                for (int j = 0; j < delta2.cols(); ++j)
                    delta2(i, j) *= reluDerivFn(z2(i, j));
            Matrix gradW2 = a1.transpose().multiply(delta2);
            Matrix gradb2 = colSum(delta2);

            Matrix delta1 = delta2.multiply(W2.transpose());
            for (int i = 0; i < delta1.rows(); ++i)
                for (int j = 0; j < delta1.cols(); ++j)
                    delta1(i, j) *= reluDerivFn(z1(i, j));
            Matrix gradW1 = input.transpose().multiply(delta1);
            Matrix gradb1 = colSum(delta1);

            // Actualizar parámetros
            W3 = W3 - gradW3 * lr;
            b3 = b3 - gradb3 * lr;
            W2 = W2 - gradW2 * lr;
            b2 = b2 - gradb2 * lr;
            W1 = W1 - gradW1 * lr;
            b1 = b1 - gradb1 * lr;
        }

        float acc = computeAccuracy(X, labels, numSamples);
        std::cout << "Epoca " << std::setw(2) << epoch
                  << " | Loss: " << std::fixed << std::setprecision(4) << epochLoss / (batches * batchSize)
                  << " | Accuracy: " << std::setprecision(2) << acc << "%" << std::endl;
    }
}

float MLP_CPU::computeAccuracy(const std::vector<float>& X,
                               const std::vector<int>& labels, int numSamples) {
    int correct = 0;
    int evalBatch = 100;
    Matrix z1, a1, z2, a2, z3;

    for (int start = 0; start < numSamples; start += evalBatch) {
        int M = std::min(evalBatch, numSamples - start);
        Matrix input(M, 784);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < 784; ++j)
                input(i, j) = X[(start + i) * 784 + j];

        Matrix a3 = forwardPass(input, z1, a1, z2, a2, z3);
        for (int i = 0; i < M; ++i) {
            int pred = 0;
            float best = a3(i, 0);
            for (int j = 1; j < 10; ++j)
                if (a3(i, j) > best) { best = a3(i, j); pred = j; }
            if (pred == labels[start + i]) ++correct;
        }
    }
    return 100.0f * correct / numSamples;
}
