#include "kernels.cuh"

// Multiplicación de matrices en GPU con tiling y shared memory
__global__ void matMulKernel(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Sumar bias a cada fila
__global__ void addBiasKernel(float* out, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    out[idx] += bias[idx % cols];
}

// Función de activación en GPU: ReLU
__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (data[idx] < 0.0f) data[idx] = 0.0f;
}

// Derivada de ReLU aplicada al gradiente
__global__ void reluBackwardKernel(float* grad, const float* preAct, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (preAct[idx] <= 0.0f) grad[idx] = 0.0f;
}

// Softmax por filas con estabilidad numérica
__global__ void softmaxKernel(float* data, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float maxVal = data[row * cols];
    for (int j = 1; j < cols; ++j)
        maxVal = fmaxf(maxVal, data[row * cols + j]);

    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        float e = expf(data[row * cols + j] - maxVal);
        data[row * cols + j] = e;
        sum += e;
    }
    for (int j = 0; j < cols; ++j)
        data[row * cols + j] /= sum;
}

// Actualizar parámetros
__global__ void updateWeightsKernel(float* weights, const float* gradW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    weights[idx] -= lr * gradW[idx];
}

// Transpuesta de matriz
__global__ void transposeKernel(const float* A, float* B, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // columna
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= cols || y >= rows) return;
    B[x * rows + y] = A[y * cols + x];
}

// Producto elemento a elemento (Hadamard)
__global__ void hadamardKernel(float* A, const float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    A[idx] *= B[idx];
}

// Calcular gradiente de bias: suma por columnas
__global__ void sumColumnsKernel(const float* delta, float* gradBias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (int i = 0; i < rows; ++i)
        sum += delta[i * cols + col];
    gradBias[col] = sum;
}

// Gradiente softmax + cross-entropy promediado por batch
__global__ void crossEntropyGradKernel(const float* predicted, const float* target,
                                       float* out, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = (predicted[idx] - target[idx]) / batch;
}
