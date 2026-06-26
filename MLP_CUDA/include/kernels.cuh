#ifndef KERNELS_CUH
#define KERNELS_CUH

#define TILE_SIZE 16

// Multiplicación de matrices en GPU: C = A * B
// A: (M x K), B: (K x N), C: (M x N)
__global__ void matMulKernel(const float* A, const float* B, float* C,
                             int M, int K, int N);

// Sumar bias a cada fila de la matriz
__global__ void addBiasKernel(float* out, const float* bias, int rows, int cols);

// Función de activación en GPU: ReLU elemento a elemento
__global__ void reluKernel(float* data, int size);

// Derivada de ReLU: out[i] = (preAct[i] > 0) ? grad[i] : 0
__global__ void reluBackwardKernel(float* grad, const float* preAct, int size);

// Softmax por filas con estabilidad numérica (max trick)
__global__ void softmaxKernel(float* data, int rows, int cols);

// Actualizar pesos: W -= lr * gradW
__global__ void updateWeightsKernel(float* weights, const float* gradW, float lr, int size);

// Transpuesta de matriz: B[j][i] = A[i][j]
__global__ void transposeKernel(const float* A, float* B, int rows, int cols);

// Producto elemento a elemento (Hadamard): A[i] = A[i] * B[i]
__global__ void hadamardKernel(float* A, const float* B, int size);

// Calcular gradiente de bias: suma por columnas del delta
__global__ void sumColumnsKernel(const float* delta, float* gradBias, int rows, int cols);

// Gradiente softmax + cross-entropy: out = (predicted - target) / batch
__global__ void crossEntropyGradKernel(const float* predicted, const float* target,
                                       float* out, int batch, int size);

#endif
