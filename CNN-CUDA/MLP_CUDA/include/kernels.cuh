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

// ===================== Kernels de convolucion 2D (CNN) =====================
// Todos operan sobre UNA imagen a la vez (sin padding, stride 1).
// Layout: tensores como [C][H][W] planos (C mas externo).
// weights layout: [Cout][Cin][K][K]

// Forward: output[oc][oy][ox] = bias[oc] + sum(input * weights)
__global__ void conv2dForwardKernel(
    const float* input, int Cin, int H, int W,
    const float* weights, const float* bias,
    float* output, int Cout, int Ho, int Wo, int K);

// Backward respecto a los pesos (acumula, no resetea; llamar fillZeros antes del batch)
__global__ void conv2dBackwardWeightsKernel(
    const float* input, int Cin, int H, int W,
    const float* dOutput, int Cout, int Ho, int Wo, int K,
    float* dWeights);

// Backward respecto al bias (acumula, no resetea)
__global__ void conv2dBackwardBiasKernel(
    const float* dOutput, int Cout, int Ho, int Wo, float* dBias);

// Backward respecto a la entrada: convolucion "full" con pesos rotados 180 grados
__global__ void conv2dBackwardInputKernel(
    const float* dOutput, int Cout, int Ho, int Wo,
    const float* weights, int Cin, int K,
    float* dInput, int H, int W);

// ===================== Kernels de max pooling 2D =====================

// Forward: guarda en argmax el indice absoluto (dentro del canal) del maximo
__global__ void maxPool2dForwardKernel(
    const float* input, int C, int H, int W,
    float* output, int* argmax, int Ho, int Wo, int pool, int stride);

// Backward: enruta el gradiente solo a la posicion ganadora (dInput debe estar en 0)
__global__ void maxPool2dBackwardKernel(
    const float* dOutput, const int* argmax, float* dInput,
    int C, int H, int W, int Ho, int Wo);

#endif
