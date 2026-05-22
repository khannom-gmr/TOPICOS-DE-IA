#include "bfs_algorithm.h"

#include <algorithm>
#include <queue>

namespace {

bool estaDentroBFS(int fila, int columna, int total_filas, int total_columnas) {
    return fila >= 0 && fila < total_filas && columna >= 0 && columna < total_columnas;
}

int aIdBFS(int fila, int columna, int total_columnas) {
    return fila * total_columnas + columna;
}

std::vector<int> reconstruirRutaBFS(const std::vector<int>& viene_de, int actual) {
    std::vector<int> ruta;
    while (actual != -1) {
        ruta.push_back(actual);
        actual = viene_de[actual];
    }
    std::reverse(ruta.begin(), ruta.end());
    return ruta;
}

}

EjecucionBFS ejecutarBFS(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal
) {
    EjecucionBFS ejecucion;

    if (cuadricula.empty()) return ejecucion;
    if (total_columnas <= 0 || total_filas <= 0) return ejecucion;
    if (id_inicio < 0 || id_fin < 0) return ejecucion;
    if (id_inicio >= static_cast<int>(cuadricula.size()) || id_fin >= static_cast<int>(cuadricula.size())) return ejecucion;
    if (cuadricula[id_inicio].bloqueado || cuadricula[id_fin].bloqueado) return ejecucion;

    std::queue<int> cola;
    std::vector<int> viene_de(cuadricula.size(), -1);
    std::vector<bool> visitado(cuadricula.size(), false);
    std::vector<bool> generado(cuadricula.size(), false);

    cola.push(id_inicio);
    visitado[id_inicio] = true;
    generado[id_inicio] = true;
    ejecucion.orden_generados.push_back(id_inicio);

    const int direcciones4[4][2] = {
        { 0,  1},
        { 1,  0},
        { 0, -1},
        {-1,  0}
    };

    const int direcciones8[8][2] = {
        { 0,  1},
        { 1,  0},
        { 0, -1},
        {-1,  0},
        { 1,  1},
        { 1, -1},
        {-1,  1},
        {-1, -1}
    };

    while (!cola.empty()) {
        int actual = cola.front();
        cola.pop();

        if (visitado[actual]) {
            // already processed
        }
        ejecucion.orden_expandidos.push_back(actual);

        if (actual == id_fin) {
            ejecucion.ruta = reconstruirRutaBFS(viene_de, id_fin);
            return ejecucion;
        }

        const NodoCuadricula& nodo_actual = cuadricula[actual];

        const int (*dirs)[2] = permitir_diagonal ? direcciones8 : direcciones4;
        const int cantidad_direcciones = permitir_diagonal ? 8 : 4;

        for (int i = 0; i < cantidad_direcciones; ++i) {
            const int nueva_fila = nodo_actual.row + dirs[i][0];
            const int nueva_columna = nodo_actual.col + dirs[i][1];

            if (!estaDentroBFS(nueva_fila, nueva_columna, total_filas, total_columnas)) {
                continue;
            }

            const int id_vecino = aIdBFS(nueva_fila, nueva_columna, total_columnas);
            if (cuadricula[id_vecino].bloqueado || visitado[id_vecino]) {
                continue;
            }

            // mark as visited when enqueued to preserve BFS layering
            visitado[id_vecino] = true;
            viene_de[id_vecino] = actual;
            cola.push(id_vecino);
            if (!generado[id_vecino]) {
                generado[id_vecino] = true;
                ejecucion.orden_generados.push_back(id_vecino);
            }
        }
    }

    return ejecucion;
}

std::vector<int> encontrarRutaBFS(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal
) {
    return ejecutarBFS(cuadricula, total_columnas, total_filas, id_inicio, id_fin, permitir_diagonal).ruta;
}
