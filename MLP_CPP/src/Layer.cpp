#include "Layer.h"
#include "Activations.h"

Layer::Layer(int inputSize, int outputSize, const std::string& activation)
    : activation_(activation) {
    // Inicializar pesos aleatorios pequeños y bias en cero
    weights = Matrix::random(inputSize, outputSize, -0.5, 0.5);
    biases = Matrix::zeros(1, outputSize);
}

// Propagación hacia adelante
Matrix Layer::forward(const Matrix& input) {
    lastInput = input;
    Matrix z = input.multiply(weights);

    // Sumar bias a cada fila
    for (int i = 0; i < z.rows(); ++i) {
        for (int j = 0; j < z.cols(); ++j) {
            z(i, j) += biases(0, j);
        }
    }
    lastPreActivation = z;

    if (activation_ == "relu") {
        return applyRelu(z);
    }
    return softmax(z);
}

// Propagación hacia atrás y actualización de parámetros
Matrix Layer::backward(const Matrix& gradient, double learningRate) {
    Matrix delta = gradient;

    // Para softmax el gradiente ya es (predicted - target); para relu se aplica la derivada
    if (activation_ == "relu") {
        Matrix deriv = applyReluDerivative(lastPreActivation);
        for (int i = 0; i < delta.rows(); ++i) {
            for (int j = 0; j < delta.cols(); ++j) {
                delta(i, j) *= deriv(i, j);
            }
        }
    }

    // Calcular gradientes de pesos y bias
    Matrix gradWeights = lastInput.transpose().multiply(delta);
    Matrix gradInput = delta.multiply(weights.transpose());

    Matrix gradBiases = Matrix::zeros(1, biases.cols());
    for (int i = 0; i < delta.rows(); ++i) {
        for (int j = 0; j < delta.cols(); ++j) {
            gradBiases(0, j) += delta(i, j);
        }
    }

    // Actualizar parámetros
    weights = weights - gradWeights * learningRate;
    biases = biases - gradBiases * learningRate;

    return gradInput;
}
