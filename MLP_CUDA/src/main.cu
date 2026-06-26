#include "MLP_CPU.h"
#include "MLP_CUDA.cuh"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cuda_runtime.h>

// Lee un entero de 4 bytes en big-endian y lo convierte a little-endian
static uint32_t readBigEndian(std::ifstream& file) {
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), 4);
    uint32_t b0 = (value & 0x000000FFu) << 24;
    uint32_t b1 = (value & 0x0000FF00u) << 8;
    uint32_t b2 = (value & 0x00FF0000u) >> 8;
    uint32_t b3 = (value & 0xFF000000u) >> 24;
    return b0 | b1 | b2 | b3;
}

// Lee imágenes MNIST en formato IDX y normaliza a [0,1]
bool loadMNISTImages(const std::string& path, std::vector<float>& images, int& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        printf("Fallo al abrir imagenes: %s (cwd=%s)\n", path.c_str(), std::filesystem::current_path().string().c_str());
        return false;
    }

    readBigEndian(file);                  // magic
    count = static_cast<int>(readBigEndian(file));
    int rows = static_cast<int>(readBigEndian(file));
    int cols = static_cast<int>(readBigEndian(file));
    int pixels = rows * cols;

    std::vector<unsigned char> buffer(static_cast<size_t>(count) * pixels);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    images.resize(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i)
        images[i] = buffer[i] / 255.0f;
    return true;
}

// Lee etiquetas MNIST en formato IDX
bool loadMNISTLabels(const std::string& path, std::vector<int>& labels, int& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        printf("Fallo al abrir etiquetas: %s (cwd=%s)\n", path.c_str(), std::filesystem::current_path().string().c_str());
        return false;
    }

    readBigEndian(file);                  // magic
    count = static_cast<int>(readBigEndian(file));

    std::vector<unsigned char> buffer(count);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    labels.resize(count);
    for (int i = 0; i < count; ++i)
        labels[i] = static_cast<int>(buffer[i]);
    return true;
}

int main() {
    const std::string dir = "data/";
    std::vector<float> trainX, testX;
    std::vector<int> trainY, testY;
    int nTrain = 0, nTest = 0, tmp = 0;

    printf("Cargando MNIST...\n");
    bool ok = loadMNISTImages(dir + "train-images.idx3-ubyte", trainX, nTrain)
            && loadMNISTLabels(dir + "train-labels.idx1-ubyte", trainY, tmp)
            && loadMNISTImages(dir + "t10k-images.idx3-ubyte", testX, nTest)
            && loadMNISTLabels(dir + "t10k-labels.idx1-ubyte", testY, tmp);

    if (!ok) {
        printf("No se encontraron los archivos MNIST en %s\n", dir.c_str());
        printf("Consulta data/README_MNIST.md para descargarlos.\n");
        return 1;
    }
    printf("  Train: %d imagenes, Test: %d imagenes\n", nTrain, nTest);

    const int epochs = 10;
    const float lr = 0.01f;
    const int batchSize = 64;

    // Entrenamiento en CPU
    printf("\n--- Entrenamiento en CPU ---\n");
    MLP_CPU cpu;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu.train(trainX, trainY, nTrain, epochs, lr, batchSize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float>(cpuEnd - cpuStart).count();
    float cpuAcc = cpu.computeAccuracy(testX, testY, nTest);
    printf("Tiempo CPU: %.2f segundos\n", cpuTime);

    // Entrenamiento en GPU
    printf("\n--- Entrenamiento en GPU ---\n");
    MLP_CUDA gpu;
    cudaEvent_t gpuStart, gpuEnd;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuEnd);
    cudaEventRecord(gpuStart);
    gpu.train(trainX, trainY, nTrain, epochs, lr, batchSize);
    cudaEventRecord(gpuEnd);
    cudaEventSynchronize(gpuEnd);
    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, gpuStart, gpuEnd);
    float gpuTime = gpuMs / 1000.0f;
    float gpuAcc = gpu.computeAccuracy(testX, testY, nTest);
    printf("Tiempo GPU: %.2f segundos\n", gpuTime);

    // Comparativa final
    printf("Comparativa CPU vs GPU\n");
    printf("CPU | Tiempo: %.2fs | Accuracy: %.2f%%\n", cpuTime, cpuAcc);
    printf("GPU | Tiempo: %.2fs | Accuracy: %.2f%%\n", gpuTime, gpuAcc);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);

    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuEnd);
    return 0;
}
