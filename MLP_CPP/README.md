# MLP_CPP

Perceptrón multicapa (MLP) implementado completamente en C++,  para clasificar el dataset Iris (3 clases). El proyecto sigue
el flujo siguiente: normalización de datos, forward pass, gradientes analíticos y descenso de gradiente (SGD por minibatches).

## Arquitectura de la red

- Entrada: 4 neuronas (sepal length, sepal width, petal length, petal width)
- Capa oculta 1: 8 neuronas + ReLU
- Capa oculta 2: 6 neuronas + ReLU
- Salida: 3 neuronas + Softmax
- Loss: Cross-Entropy
- Optimizador: SGD (minibatch)

Hiperparámetros: 500 épocas, learning rate 0.01, batch size 16.

## Cómo compilar

### Con make

```bash
make
```

### Con cmake

```bash
cmake -S . -B build
cmake --build build
```

## Cómo ejecutar

```bash
./mlp_iris
```

> El programa lee `data/iris.csv` con ruta relativa, por lo que debe ejecutarse
> desde la raíz del proyecto.

## Ejemplo de salida esperada

```
Epoca  10 | Loss: 1.0905 | Accuracy: 49.17%
Epoca  20 | Loss: 1.0836 | Accuracy: 68.33%
...
Epoca 500 | Loss: 0.1220 | Accuracy: 96.67%

[Test] Accuracy: 93.33%
```

## Descripción de cada archivo

| Archivo | Descripción |
|---------|-------------|
| `include/Matrix.h` / `src/Matrix.cpp` | Clase `Matrix` (row-major) con suma, resta, producto matricial, transpuesta, `apply` y fábricas `zeros`/`random`. |
| `include/Activations.h` / `src/Activations.cpp` | Funciones de activación: ReLU, su derivada y softmax por filas. |
| `include/Loss.h` / `src/Loss.cpp` | Cross-Entropy loss y su gradiente combinado con softmax. |
| `include/Layer.h` / `src/Layer.cpp` | Clase `Layer` con `forward` y `backward`; guarda pesos, bias y estados intermedios. |
| `include/MLP.h` / `src/MLP.cpp` | Clase `MLP`: encadena capas, entrena con SGD y calcula accuracy. |
| `src/main.cpp` | Punto de entrada: carga, normaliza, divide datos, codifica one-hot, entrena y evalúa. |
| `data/iris.csv` | Dataset Iris completo (150 muestras). |
| `Makefile` / `CMakeLists.txt` | Sistemas de compilación. |
