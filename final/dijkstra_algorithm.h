#ifndef DIJKSTRA_ALGORITHM_H
#define DIJKSTRA_ALGORITHM_H

#include <vector>

struct NodoCuadricula;

struct EjecucionDijkstra {
    std::vector<int> ruta;
    std::vector<int> orden_generados;
    std::vector<int> orden_expandidos;
};

EjecucionDijkstra ejecutarDijkstra(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal = true
);

std::vector<int> encontrarRutaDijkstra(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal = true
);

#endif
