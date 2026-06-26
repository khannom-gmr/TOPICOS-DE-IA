#ifndef MLP_CPU_H
#define MLP_CPU_H

#include "Matrix.h"
#include <vector>

// MLP de referencia en CPU (arquitectura 784-128-64-10) usado para comparar contra la GPU
class MLP_CPU {
public:
    MLP_CPU();

    void train(const std::vector<float>& X, const std::vector<int>& labels,
               int numSamples, int epochs, float lr, int batchSize);
    float computeAccuracy(const std::vector<float>& X,
                          const std::vector<int>& labels, int numSamples);

private:
    Matrix W1, b1, W2, b2, W3, b3;

    Matrix forwardPass(const Matrix& input, Matrix& z1, Matrix& a1,
                       Matrix& z2, Matrix& a2, Matrix& z3) const;
    void initWeights();
};

#endif
