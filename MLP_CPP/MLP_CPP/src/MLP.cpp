#include "MLP.h"
#include "Loss.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

MLP::MLP(std::vector<Layer> layers) : layers_(std::move(layers)) {}

// Propagación hacia adelante por todas las capas
Matrix MLP::forwardPass(const Matrix& input) {
    Matrix out = input;
    for (auto& layer : layers_) {
        out = layer.forward(out);
    }
    return out;
}

Matrix MLP::predict(const Matrix& input) {
    return forwardPass(input);
}

// Propagación hacia atrás por todas las capas
void MLP::backpropagation(const Matrix& gradient, double learningRate) {
    Matrix grad = gradient;
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        grad = layers_[i].backward(grad, learningRate);
    }
}

void MLP::train(const Matrix& X, const Matrix& y, int epochs, double learningRate, int batchSize) {
    int samples = X.rows();
    std::vector<int> indices(samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(42);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int start = 0; start < samples; start += batchSize) {
            int end = std::min(start + batchSize, samples);
            int n = end - start;

            // Construir el minibatch
            Matrix batchX(n, X.cols());
            Matrix batchY(n, y.cols());
            for (int i = 0; i < n; ++i) {
                int idx = indices[start + i];
                for (int j = 0; j < X.cols(); ++j) batchX(i, j) = X(idx, j);
                for (int j = 0; j < y.cols(); ++j) batchY(i, j) = y(idx, j);
            }

            Matrix predicted = forwardPass(batchX);
            Matrix grad = crossEntropyGradient(predicted, batchY);
            backpropagation(grad, learningRate);
        }

        if (epoch % 10 == 0) {
            Matrix predicted = forwardPass(X);
            double loss = crossEntropyLoss(predicted, y);
            double acc = computeAccuracy(X, y);
            std::cout << "Epoca " << std::setw(3) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss
                      << " | Accuracy: " << std::setprecision(2) << acc << "%" << std::endl;
        }
    }
}

double MLP::computeAccuracy(const Matrix& X, const Matrix& y) {
    Matrix predicted = predict(X);
    int correct = 0;
    for (int i = 0; i < predicted.rows(); ++i) {
        int predClass = 0, trueClass = 0;
        double maxPred = predicted(i, 0), maxTrue = y(i, 0);
        for (int j = 1; j < predicted.cols(); ++j) {
            if (predicted(i, j) > maxPred) { maxPred = predicted(i, j); predClass = j; }
            if (y(i, j) > maxTrue) { maxTrue = y(i, j); trueClass = j; }
        }
        if (predClass == trueClass) ++correct;
    }
    return 100.0 * correct / predicted.rows();
}
