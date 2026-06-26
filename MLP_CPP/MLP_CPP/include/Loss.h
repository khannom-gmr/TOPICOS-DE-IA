#ifndef LOSS_H
#define LOSS_H

#include "Matrix.h"

double crossEntropyLoss(const Matrix& predicted, const Matrix& target);
Matrix crossEntropyGradient(const Matrix& predicted, const Matrix& target);

#endif
