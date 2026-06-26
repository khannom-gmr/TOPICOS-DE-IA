#ifndef MLP_H
#define MLP_H

#include "Matrix.h"
#include "Layer.h"
#include <vector>

class MLP {
public:
    MLP(std::vector<Layer> layers);

    Matrix predict(const Matrix& input);
    void train(const Matrix& X, const Matrix& y, int epochs, double learningRate, int batchSize);
    double computeAccuracy(const Matrix& X, const Matrix& y);

private:
    std::vector<Layer> layers_;

    Matrix forwardPass(const Matrix& input);
    void backpropagation(const Matrix& gradient, double learningRate);
};

#endif
