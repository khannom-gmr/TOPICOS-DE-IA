#include "Loss.h"
#include <cmath>

// Calcular error promedio con cross-entropy
double crossEntropyLoss(const Matrix& predicted, const Matrix& target) {
    double loss = 0.0;
    int batch = predicted.rows();
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < predicted.cols(); ++j) {
            if (target(i, j) > 0.0) {
                loss -= target(i, j) * std::log(predicted(i, j) + 1e-12);
            }
        }
    }
    return loss / batch;
}

// Gradiente softmax + cross-entropy: (predicted - target) promediado por batch
Matrix crossEntropyGradient(const Matrix& predicted, const Matrix& target) {
    int batch = predicted.rows();
    Matrix grad(predicted.rows(), predicted.cols());
    for (int i = 0; i < predicted.rows(); ++i) {
        for (int j = 0; j < predicted.cols(); ++j) {
            grad(i, j) = (predicted(i, j) - target(i, j)) / batch;
        }
    }
    return grad;
}
