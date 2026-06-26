#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "Matrix.h"

// Función de activación
double relu(double x);
double reluDerivative(double x);

Matrix applyRelu(const Matrix& m);
Matrix applyReluDerivative(const Matrix& m);

// Softmax por filas (cada fila es una muestra)
Matrix softmax(const Matrix& x);

#endif
