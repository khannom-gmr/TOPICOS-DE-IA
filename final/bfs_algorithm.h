#ifndef BFS_ALGORITHM_H
#define BFS_ALGORITHM_H

#include <vector>
#include "astar_algorithm.h"

struct EjecucionBFS {
    std::vector<int> ruta;
    std::vector<int> orden_generados;
    std::vector<int> orden_expandidos;
};

EjecucionBFS ejecutarBFS(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal = true
);

std::vector<int> encontrarRutaBFS(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal = true
);

#endif
