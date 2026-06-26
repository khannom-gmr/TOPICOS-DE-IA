# MLP_CUDA

Perceptrón multicapa (MLP) para el reconocimiento de dígitos manuscritos del dataset
MNIST, implementado en CUDA C++. El proyecto incluye dos entrenadores con la misma
arquitectura —uno en CPU y otro en GPU— y reporta una comparativa de tiempo y precisión
entre ambos para cuantificar el speedup obtenido al paralelizar en GPU.

## Arquitectura de la red

| Capa            | Tamaño | Activación |
|-----------------|--------|------------|
| Entrada         | 784    | —          |
| Oculta 1        | 128    | ReLU       |
| Oculta 2        | 64     | ReLU       |
| Salida          | 10     | Softmax    |

- **Loss:** Cross-Entropy
- **Optimizador:** SGD por minibatches
- **Hiperparámetros:** 10 épocas, learning rate 0.01, batch size 64
- Los pesos se inicializan con Xavier uniforme (semilla fija 42) y se transfieren a la GPU.

Todos los kernels (multiplicación de matrices con tiling en shared memory, ReLU, softmax,
transpuesta, actualización de pesos, etc.) están implementados manualmente; no se usa cuBLAS.

## Requisitos del sistema

- **CUDA Toolkit 12.x** (incluye `nvcc`)
- **g++** con soporte C++17
- **CMake 3.18+** (opcional, alternativa al Makefile)
- GPU NVIDIA con capacidad de cómputo compatible. El proyecto está configurado para
  `sm_89` (Ada Lovelace, p. ej. RTX 4070 Ti Super). Ajusta el valor si tu GPU es distinta.


## Verificar que la GPU es detectada y que esta instalado el nvcc

```bash
nvidia-smi        # muestra la GPU, el driver y la versión de CUDA soportada
nvcc --version    # muestra la versión del compilador CUDA instalado
```

## Compilar

### Con make

```bash
make
```

### Con cmake

```bash
cmake -S . -B build
cmake --build build
```

## Ejecutar

```bash
./mlp_mnist
```

> Ejecuta desde la raíz del proyecto para que encuentre los archivos en `data/`.

## Ejemplo de salida esperada

```
Cargando MNIST...
  Train: 60000 imagenes, Test: 10000 imagenes

--- Entrenamiento en CPU ---
Epoca  1 | Loss: 1.8432 | Accuracy: 62.13%
...
Epoca 10 | Loss: 0.3821 | Accuracy: 93.45%
Tiempo CPU: 47.23 segundos

--- Entrenamiento en GPU ---
Epoca  1 | Loss: 1.8401 | Accuracy: 62.87%
...
Epoca 10 | Loss: 0.3794 | Accuracy: 93.89%
Tiempo GPU: 3.91 segundos

   Comparativa CPU vs GPU
CPU | Tiempo: 47.23s | Accuracy: 93.45%
GPU | Tiempo:  3.91s | Accuracy: 93.89%
Speedup: 12.08x
```

> Los tiempos exactos y el speedup dependen del hardware (CPU y GPU concretas).

## Guardar y cargar pesos

Los pesos viven en memoria del device como `float`. Para persistirlos, cópialos al host
con `copyToHost` y escríbelos en un archivo binario; para restaurarlos, léelos y súbelos
con `copyFromHost`. Esquema básico:

```cpp
// Guardar
std::vector<float> buffer(rows * cols);
weights.copyToHost(buffer.data(), buffer.size());
std::ofstream out("weights.bin", std::ios::binary);
out.write(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));

// Cargar
std::ifstream in("weights.bin", std::ios::binary);
in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
weights.copyFromHost(buffer.data(), buffer.size());
```

Repite el proceso para `W1, b1, W2, b2, W3, b3` (en el mismo orden al guardar y cargar).

## Descripción de los archivos

| Archivo | Descripción |
|---------|-------------|
| `include/Matrix.h` / `src/Matrix.cpp` | Clase `Matrix` row-major (float) usada por el MLP de CPU. |
| `include/MLP_CPU.h` / `src/MLP_CPU.cpp` | MLP de referencia en CPU (forward, backward y SGD). |
| `include/kernels.cuh` / `src/kernels.cu` | Kernels CUDA: matmul con tiling, bias, ReLU, softmax, transpuesta, Hadamard, actualización de pesos y gradientes. |
| `include/CudaMatrix.cuh` / `src/CudaMatrix.cu` | Wrapper RAII de memoria en device (move-only, sin fugas). |
| `include/MLP_CUDA.cuh` / `src/MLP_CUDA.cu` | MLP en GPU: mantiene pesos y activaciones en device y orquesta forward/backward. |
| `src/main.cu` | Lector de MNIST (IDX), entrenamiento CPU y GPU, medición de tiempos y comparativa. |
| `data/README_MNIST.md` | Instrucciones para descargar el dataset. |
| `Makefile` / `CMakeLists.txt` | Sistemas de compilación. |

## Crear rama git

```bash
git checkout -b mlp-cuda
git add .
git commit -m "Implementación MLP en CUDA con comparativa CPU vs GPU"
git push origin mlp-cuda
```
