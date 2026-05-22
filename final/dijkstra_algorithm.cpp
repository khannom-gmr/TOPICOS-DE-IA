#include "dijkstra_algorithm.h"

#include "astar_algorithm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace {

constexpr float kInfinito = std::numeric_limits<float>::infinity();

bool estaDentro(int fila, int columna, int total_filas, int total_columnas) {
    return fila >= 0 && fila < total_filas && columna >= 0 && columna < total_columnas;
}

int aId(int fila, int columna, int total_columnas) {
    return fila * total_columnas + columna;
}

std::vector<int> reconstruirRuta(const std::vector<int>& viene_de, int actual) {
    std::vector<int> ruta;
    while (actual != -1) {
        ruta.push_back(actual);
        actual = viene_de[actual];
    }
    std::reverse(ruta.begin(), ruta.end());
    return ruta;
}

}

EjecucionDijkstra ejecutarDijkstra(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal
) {
    EjecucionDijkstra ejecucion;

    if (cuadricula.empty()) return ejecucion;
    if (total_columnas <= 0 || total_filas <= 0) return ejecucion;
    if (id_inicio < 0 || id_fin < 0) return ejecucion;
    if (id_inicio >= static_cast<int>(cuadricula.size()) || id_fin >= static_cast<int>(cuadricula.size())) return ejecucion;
    if (cuadricula[id_inicio].bloqueado || cuadricula[id_fin].bloqueado) return ejecucion;

    struct NodoCola {
        int id;
        float distancia;
    };

    auto comparar = [](const NodoCola& a, const NodoCola& b) {
        return a.distancia > b.distancia;
    };

    std::priority_queue<NodoCola, std::vector<NodoCola>, decltype(comparar)> cola_prioridad(comparar);

    std::vector<float> distancia(cuadricula.size(), kInfinito);
    std::vector<int> viene_de(cuadricula.size(), -1);
    std::vector<bool> cerrado(cuadricula.size(), false);
    std::vector<bool> generado(cuadricula.size(), false);

    distancia[id_inicio] = 0.0f;
    cola_prioridad.push({id_inicio, 0.0f});
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

    while (!cola_prioridad.empty()) {
        NodoCola actual = cola_prioridad.top();
        cola_prioridad.pop();

        if (cerrado[actual.id]) {
            continue;
        }
        cerrado[actual.id] = true;
        ejecucion.orden_expandidos.push_back(actual.id);

        if (actual.id == id_fin) {
            ejecucion.ruta = reconstruirRuta(viene_de, id_fin);
            return ejecucion;
        }

        const NodoCuadricula& nodo_actual = cuadricula[actual.id];

        const int (*dirs)[2] = permitir_diagonal ? direcciones8 : direcciones4;
        const int cantidad_direcciones = permitir_diagonal ? 8 : 4;

        for (int i = 0; i < cantidad_direcciones; i++) {
            const int nueva_fila = nodo_actual.row + dirs[i][0];
            const int nueva_columna = nodo_actual.col + dirs[i][1];

            if (!estaDentro(nueva_fila, nueva_columna, total_filas, total_columnas)) {
                continue;
            }

            const int id_vecino = aId(nueva_fila, nueva_columna, total_columnas);
            if (cuadricula[id_vecino].bloqueado || cerrado[id_vecino]) {
                continue;
            }

            const bool es_diagonal = dirs[i][0] != 0 && dirs[i][1] != 0;
            const float costo_paso = es_diagonal ? std::sqrt(2.0f) : 1.0f;
            const float distancia_tentativa = distancia[actual.id] + costo_paso;

            if (distancia_tentativa < distancia[id_vecino]) {
                viene_de[id_vecino] = actual.id;
                distancia[id_vecino] = distancia_tentativa;
                cola_prioridad.push({id_vecino, distancia[id_vecino]});
                if (!generado[id_vecino]) {
                    generado[id_vecino] = true;
                    ejecucion.orden_generados.push_back(id_vecino);
                }
            }
        }
    }

    return ejecucion;
}

std::vector<int> encontrarRutaDijkstra(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal
) {
    return ejecutarDijkstra(cuadricula, total_columnas, total_filas, id_inicio, id_fin, permitir_diagonal).ruta;
}
