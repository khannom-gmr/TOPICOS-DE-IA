#ifndef ASTAR_ALGORITHM_H
#define ASTAR_ALGORITHM_H

#include <vector>

struct NodoCuadricula {
    int id;
    int col;
    int row;
    bool bloqueado;
};

struct EjecucionAEstrella {
    std::vector<int> ruta;
    std::vector<int> orden_generados;
    std::vector<int> orden_expandidos;
};

EjecucionAEstrella ejecutarAEstrella(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal = true
);

std::vector<int> encontrarRutaAEstrella(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal = true
);

#endif