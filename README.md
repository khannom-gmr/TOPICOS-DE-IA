#  Algoritmos de Búsqueda y Optimización

Proyecto en **C++** que implementa y visualiza algoritmos de búsqueda en grafos y resolución del **Problema del Viajero (TSP)** mediante un Algoritmo Genético.

---

##  Estructura general

```
├── asterisco.cpp / .h          # Lógica visual, cuadrícula, animación, callbacks
├── astar_algorithm.cpp / .h    # Algoritmo A* con heurística Octile
├── dijkstra_algorithm.cpp / .h # Dijkstra (reutiliza estructuras de A*)
├── genetic_algorithm.cpp / .h  # Clase AlgoritmoGenetico (ciclo evolutivo TSP)
└── main.cpp                    # Punto de entrada
```

---

##  Módulo 1 — Visualizador de Búsqueda en Grafos

Aplicación interactiva con **OpenGL** que compara A\*, Dijkstra y BFS sobre una cuadrícula configurable con obstáculos. Incluye animación paso a paso de nodos generados, expandidos y ruta final.

### Algoritmos

| Algoritmo | Heurística | Nodos explorados | Óptimo | Considera pesos |
|-----------|-----------|-----------------|--------|----------------|
| **A\***   | ✅ Octile Distance | Menos | ✅ | ✅ |
| **Dijkstra** | ❌ | Más | ✅ | ✅ |
| **BFS**   | ❌ | Todos | ⚠️ Solo por pasos | ❌ |

#### A\*
Usa `f(n) = g(n) + h(n)`. El costo `g` es el acumulado desde el origen y `h` es la heurística Octile Distance, admisible con movimientos diagonales. Es el más eficiente de los tres.

```
h = dx + dy + (√2 − 2) · min(dx, dy)
```

#### Dijkstra
Expande siempre el nodo de menor costo acumulado sin guía direccional. Garantiza la ruta óptima pero explora más nodos que A\*.

#### BFS
Usa una cola **FIFO** y explora por niveles. No considera pesos — cada arista vale 1. Útil como base conceptual y cuando solo importa minimizar el número de saltos.

### Leyenda visual

| Color | Significado |
|-------|-------------|
| 🔵 Azul | Nodo normal |
| 🟣 Morado | Generado (frontera) |
| 🟡 Amarillo | Expandido |
| 🟢 Verde | En ruta final |
| 🔴 Rojo | Obstáculo eliminado |
| 🟠 Naranja | Nodo inicio |
| 🩵 Cyan | Nodo destino |

### Controles

| Tecla | Acción |
|-------|--------|
| `R` | Reiniciar cuadrícula |
| `D` | Alternar A\* / Dijkstra |
| `O` | Eliminar nodos aleatorios |
| `C` | Limpiar obstáculos |
| `Space` | Recalcular ruta |
| `Click` | Toggle obstáculo en nodo |
| `ESC` | Salir |

---

##  Módulo 2 — TSP con Algoritmo Genético

Solucionador por línea de comandos del **Problema del Viajero** sobre 8 ciudades peruanas. El TSP es NP-difícil: el algoritmo genético no garantiza la solución óptima, pero encuentra soluciones muy buenas en tiempo razonable.

### Ciudades del ejemplo

```
[0] Lima      [1] Arequipa  [2] Cusco     [3] Trujillo
[4] Chiclayo  [5] Iquitos   [6] Puno      [7] Tacna
```

### Operadores genéticos

| Operador | Implementación |
|----------|---------------|
| **Inicialización** | Permutaciones aleatorias con `iota` + `shuffle` |
| **Aptitud** | `−distanciaTotal(ruta)` — mayor aptitud = ruta más corta |
| **Selección** | Torneo de 3 candidatos aleatorios |
| **Cruce** | Order Crossover (OX) — garantiza permutaciones válidas |
| **Mutación** | Swap de dos posiciones con probabilidad `tasaMutacion_` |
| **Elitismo** | El mejor individuo se conserva intacto cada generación |

#### Cruce OX explicado

```
Padre A:  [0, 1, 2 | 3, 4 | 5, 6]   ← segmento central copiado
Padre B:  [4, 5, 2,  0, 3,  1, 6]   ← resto tomado en orden
Hijo:     [5, 2, 0 | 3, 4 | 1, 6]   ← permutación válida
```

### Uso

```bash
./tsp_genetico [opciones]

  --poblacion    <N>   Tamaño de población     (default: 200)
  --mutacion     <R>   Tasa de mutación [0,1]  (default: 0.02)
  --generaciones <N>   Máximo de generaciones  (default: 10000)
```

### Ejemplo de salida

```
=== TSP con Algoritmo Genetico ===
Ciudades      : 8
Poblacion     : 200
Mutacion      : 0.02
Generaciones  : 10000

Mejor ruta encontrada:
  Lima -> Trujillo -> Chiclayo -> Iquitos -> Cusco -> Puno -> Arequipa -> Tacna -> Lima

Distancia total: 6487 km
```

---

##  Dependencias

- C++17
- OpenGL / GLFW — módulo de visualización

---

##  Compilación

```bash
# Módulo de visualización
g++ -std=c++17 main.cpp asterisco.cpp astar_algorithm.cpp dijkstra_algorithm.cpp -lglfw -lGL -o visualizador

# Módulo TSP
g++ -std=c++17 main.cpp genetic_algorithm.cpp -o tsp_genetico
```


---

## Modulo 3 — Vision Transformer (ViT) para MNIST en C++ y CUDA

Implementacion desde cero de un **Vision Transformer (ViT)** para clasificar los 10 digitos del dataset MNIST. Sin frameworks de deep learning - solo STL de C++, cuBLAS y kernels CUDA propios.

### Estructura del Modulo ViT

```
├── CMakeLists.txt          # Raiz: incluye vit_cpp y vit_cuda
├── data/
│   └── download_mnist.py   # Script para descargar MNIST automaticamente
├── vit_cpp/                # Proyecto C++ CPU independiente
│   ├── include/            # Encabezados (tensor, layers, attention, transformer)
│   ├── src/                # Implementacion C++ puro
│   ├── apps/               # Ejecutables (train_cpu.cpp, inference.cpp)
│   └── tests/              # Pruebas unitarias (41/41 pasadas)
└── vit_cuda/               # Proyecto CUDA GPU independiente
    ├── include/            # Encabezados CUDA (cuda_tensor, kernels, etc.)
    ├── src/                # Kernels CUDA y clases GPU (.cu)
    └── apps/               # Ejecutable (train_gpu.cpp)
```

### Arquitectura ViT

```
[B, 784]  ->  PatchEmbed(7x7, 16 parches)  ->  Linear [49->64]
          ->  + CLS token + pos_embed        ->  [B, 17, 64]
          ->  4x TransformerBlock            ->  [B, 17, 64]
               L- LayerNorm -> MHSA (4 cabezas, head_dim=16)
               L- LayerNorm -> MLP (64->128->GELU->64)
          ->  CLS[:, 0, :]                   ->  [B, 64]
          ->  LayerNorm -> Linear(64->10)    ->  [B, 10]
```

### Instrucciones de Compilacion y Ejecucion

#### 1. Descargar MNIST
```bash
python data/download_mnist.py
```

#### 2. Compilar con CMake
```bash
cmake -B build
cmake --build build --config Release
```

#### 3. Entrenar en GPU (CUDA)
```bash
./build/bin/Release/vit_cuda.exe --datos ./data --epocas 10 --lote 128
```

#### 4. Entrenar en CPU
```bash
./build/bin/Release/vit_cpu.exe --datos ./data --epocas 10 --lote 64
```

#### 5. Inferencia y Matriz de Confusion
```bash
./build/bin/Release/vit_infer.exe --checkpoint ./checkpoints/vit_gpu_epoca10.bin --datos ./data
```

#### 6. Pruebas Unitarias
```bash
./build/bin/Release/test_tensor.exe
./build/bin/Release/test_attention.exe
./build/bin/Release/test_mnist.exe ./data
```

### Resultados Reales (GPU — RTX 4070 Ti Super)

```
Epoca [ 1/10]  perdida=0.8928  exactitud=70.32%  tiempo=84.5s
Epoca [ 2/10]  perdida=0.2798  exactitud=91.34%  tiempo=85.8s
Epoca [ 3/10]  perdida=0.1406  exactitud=95.81%  tiempo=83.9s
==> Prueba  perdida=0.1361  exactitud=95.74%
```
