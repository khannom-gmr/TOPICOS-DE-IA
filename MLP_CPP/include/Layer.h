#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include <string>

class Layer {
public:
    Layer(int inputSize, int outputSize, const std::string& activation);

    // Propagación hacia adelante
    Matrix forward(const Matrix& input);
    // Propagación hacia atrás, retorna la gradient para la capa anterior
    Matrix backward(const Matrix& gradient, double learningRate);

    Matrix weights;
    Matrix biases;

private:
    std::string activation_;
    Matrix lastInput;
    Matrix lastPreActivation;
};

#endif
