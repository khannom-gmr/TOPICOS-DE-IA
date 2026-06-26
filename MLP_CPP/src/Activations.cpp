#include "Activations.h"
#include <cmath>
#include <algorithm>

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

Matrix applyRelu(const Matrix& m) {
    return m.apply(relu);
}

Matrix applyReluDerivative(const Matrix& m) {
    return m.apply(reluDerivative);
}

// Softmax por filas con estabilidad numérica
Matrix softmax(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        double maxVal = x(i, 0);
        for (int j = 1; j < x.cols(); ++j) {
            maxVal = std::max(maxVal, x(i, j));
        }
        double sum = 0.0;
        for (int j = 0; j < x.cols(); ++j) {
            double e = std::exp(x(i, j) - maxVal);
            result(i, j) = e;
            sum += e;
        }
        for (int j = 0; j < x.cols(); ++j) {
            result(i, j) /= sum;
        }
    }
    return result;
}
