Algoritmos de Búsqueda y Optimización
Proyecto en C++ que implementa y visualiza algoritmos de búsqueda en grafos y resolución del Problema del Viajero (TSP) mediante un Algoritmo Genético.

Módulo 1 — Visualizador de Búsqueda en Grafos
Aplicación interactiva con OpenGL que permite comparar A*, Dijkstra y BFS sobre una cuadrícula configurable con obstáculos.
Algoritmos implementados
A* — Búsqueda informada. Usa la función f(n) = g(n) + h(n) donde h es la distancia Octile. Es el más eficiente de los tres: la heurística guía la búsqueda hacia el destino y reduce los nodos explorados. Garantiza la ruta óptima.
Dijkstra — Búsqueda no informada. Expande siempre el nodo de menor costo acumulado sin heurística. Explora más nodos que A* pero sigue garantizando la ruta óptima en grafos con pesos no negativos.
BFS — Búsqueda en amplitud. Usa una cola FIFO y explora por niveles. No considera pesos — cada arista vale 1. Garantiza el camino con menor número de saltos, no el de menor costo.
Controles
TeclaAcciónRReiniciar cuadrículaDAlternar A* / DijkstraOEliminar nodos aleatoriosCLimpiar obstáculosSpaceRecalcular rutaClickToggle obstáculo en nodoESCSalir
Estructura del código
main.cpp                  → Punto de entrada
asterisco.cpp/h           → Lógica visual, cuadrícula, animación, callbacks
astar_algorithm.cpp/h     → Algoritmo A* con heurística Octile
dijkstra_algorithm.cpp/h  → Dijkstra (reutiliza estructuras de A*)

Módulo 2 — TSP con Algoritmo Genético
Solucionador por línea de comandos del Problema del Viajero sobre 8 ciudades peruanas.
¿Qué es el TSP?
Encontrar la ruta más corta que visite cada ciudad exactamente una vez y regrese al origen. Es un problema NP-difícil: el genético no garantiza la solución óptima, pero encuentra soluciones muy buenas en tiempo razonable.
Operadores genéticos

Inicialización — Permutaciones aleatorias válidas de las ciudades usando iota + shuffle.
Aptitud — Negativo de la distancia total. Mayor aptitud = ruta más corta.
Selección — Torneo de 3: se eligen 3 individuos al azar y pasa el de mayor aptitud.
Cruce OX — Order Crossover: copia un segmento del Padre A y rellena el resto con el orden del Padre B. Garantiza permutaciones válidas sin ciudades repetidas.
Mutación — Swap de dos posiciones aleatorias con probabilidad tasaMutacion_.
Elitismo — El mejor individuo de cada generación se conserva intacto en la siguiente.

Uso
bash./tsp_genetico [opciones]

  --poblacion   <N>   Tamaño de población     (default: 200)
  --mutacion    <R>   Tasa de mutación [0,1]  (default: 0.02)
  --generaciones <N>  Máximo generaciones     (default: 10000)
Estructura del código
main.cpp                  → Configuración CLI y matriz de distancias
genetic_algorithm.cpp/h   → Clase AlgoritmoGenetico (todo el ciclo evolutivo)

Dependencias

C++17
OpenGL / GLFW (módulo de visualización)
