# 🔍 Algoritmos de Búsqueda y Optimización

Proyecto en **C++** que implementa y visualiza algoritmos de búsqueda en grafos y resolución del **Problema del Viajero (TSP)** mediante un Algoritmo Genético.

---

## 📁 Estructura general

```
├── asterisco.cpp / .h          # Lógica visual, cuadrícula, animación, callbacks
├── astar_algorithm.cpp / .h    # Algoritmo A* con heurística Octile
├── dijkstra_algorithm.cpp / .h # Dijkstra (reutiliza estructuras de A*)
├── genetic_algorithm.cpp / .h  # Clase AlgoritmoGenetico (ciclo evolutivo TSP)
└── main.cpp                    # Punto de entrada
```

---

## 🗺️ Módulo 1 — Visualizador de Búsqueda en Grafos

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

## 🧬 Módulo 2 — TSP con Algoritmo Genético

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

## ⚙️ Dependencias

- C++17
- OpenGL / GLFW — módulo de visualización

---

## 🏗️ Compilación

```bash
# Módulo de visualización
g++ -std=c++17 main.cpp asterisco.cpp astar_algorithm.cpp dijkstra_algorithm.cpp -lglfw -lGL -o visualizador

# Módulo TSP
g++ -std=c++17 main.cpp genetic_algorithm.cpp -o tsp_genetico
```
