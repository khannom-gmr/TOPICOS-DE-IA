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

// ===================== Convolucion 2D (una imagen) =====================

// Cada thread calcula un pixel (oy,ox) del canal de salida oc
__global__ void conv2dForwardKernel(
    const float* input, int Cin, int H, int W,
    const float* weights, const float* bias,
    float* output, int Cout, int Ho, int Wo, int K) {

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    if (ox >= Wo || oy >= Ho || oc >= Cout) return;

    float sum = bias[oc];
    for (int ic = 0; ic < Cin; ++ic) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                int iy = oy + ky;
                int ix = ox + kx;
                float inVal = input[(ic * H + iy) * W + ix];
                float wVal  = weights[((oc * Cin + ic) * K + ky) * K + kx];
                sum += inVal * wVal;
            }
        }
    }
    output[(oc * Ho + oy) * Wo + ox] = sum;
}

// Un thread por cada peso (oc,ic,ky,kx); recorre todo el mapa de salida y ACUMULA
__global__ void conv2dBackwardWeightsKernel(
    const float* input, int Cin, int H, int W,
    const float* dOutput, int Cout, int Ho, int Wo, int K,
    float* dWeights) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin * K * K;
    if (idx >= total) return;

    int kx = idx % K;
    int ky = (idx / K) % K;
    int ic = (idx / (K * K)) % Cin;
    int oc = idx / (K * K * Cin);

    float sum = 0.0f;
    for (int oy = 0; oy < Ho; ++oy) {
        for (int ox = 0; ox < Wo; ++ox) {
            int iy = oy + ky;
            int ix = ox + kx;
            sum += input[(ic * H + iy) * W + ix] *
                   dOutput[(oc * Ho + oy) * Wo + ox];
        }
    }
    dWeights[idx] += sum; // acumula sobre el batch
}

// Un thread por canal de salida: suma todo el mapa dOutput de ese canal
__global__ void conv2dBackwardBiasKernel(
    const float* dOutput, int Cout, int Ho, int Wo, float* dBias) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= Cout) return;
    float sum = 0.0f;
    for (int i = 0; i < Ho * Wo; ++i) sum += dOutput[oc * Ho * Wo + i];
    dBias[oc] += sum; // acumula sobre el batch
}

// Convolucion "full": para cada pixel de entrada, suma las contribuciones de
// todos los outputs que lo usaron, con el kernel rotado 180 grados
__global__ void conv2dBackwardInputKernel(
    const float* dOutput, int Cout, int Ho, int Wo,
    const float* weights, int Cin, int K,
    float* dInput, int H, int W) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z;
    if (ix >= W || iy >= H || ic >= Cin) return;

    float sum = 0.0f;
    for (int oc = 0; oc < Cout; ++oc) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                int oy = iy - ky;
                int ox = ix - kx;
                if (oy >= 0 && oy < Ho && ox >= 0 && ox < Wo) {
                    float wVal = weights[((oc * Cin + ic) * K + ky) * K + kx];
                    sum += wVal * dOutput[(oc * Ho + oy) * Wo + ox];
                }
            }
        }
    }
    dInput[(ic * H + iy) * W + ix] = sum;
}

// ===================== Max pooling 2D (una imagen) =====================

__global__ void maxPool2dForwardKernel(
    const float* input, int C, int H, int W,
    float* output, int* argmax, int Ho, int Wo, int pool, int stride) {

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z;
    if (ox >= Wo || oy >= Ho || c >= C) return;

    float best = -1e30f;
    int bestIdx = -1;
    for (int py = 0; py < pool; ++py) {
        for (int px = 0; px < pool; ++px) {
            int iy = oy * stride + py;
            int ix = ox * stride + px;
            int idx = (c * H + iy) * W + ix;
            if (input[idx] > best) { best = input[idx]; bestIdx = idx; }
        }
    }
    int outIdx = (c * Ho + oy) * Wo + ox;
    output[outIdx] = best;
    argmax[outIdx] = bestIdx;
}

__global__ void maxPool2dBackwardKernel(
    const float* dOutput, const int* argmax, float* dInput,
    int C, int H, int W, int Ho, int Wo) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * Ho * Wo;
    if (idx >= total) return;
    atomicAdd(&dInput[argmax[idx]], dOutput[idx]);
}
