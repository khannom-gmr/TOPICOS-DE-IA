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

float heuristica(const NodoCuadricula& a, const NodoCuadricula& b, bool permitir_diagonal) {
    const float dx = static_cast<float>(std::abs(a.col - b.col));
    const float dy = static_cast<float>(std::abs(a.row - b.row));

    if (!permitir_diagonal) {
        return dx + dy;
    }

    const float d1 = 1.0f;
    const float d2 = std::sqrt(2.0f);
    return d1 * (dx + dy) + (d2 - 2.0f * d1) * std::min(dx, dy);
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

EjecucionAEstrella ejecutarAEstrella(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal
) {
    EjecucionAEstrella ejecucion;

    if (cuadricula.empty()) return ejecucion;
    if (total_columnas <= 0 || total_filas <= 0) return ejecucion;
    if (id_inicio < 0 || id_fin < 0) return ejecucion;
    if (id_inicio >= static_cast<int>(cuadricula.size()) || id_fin >= static_cast<int>(cuadricula.size())) return ejecucion;
    if (cuadricula[id_inicio].bloqueado || cuadricula[id_fin].bloqueado) return ejecucion;

    struct NodoCola {
        int id;
        float puntaje_f;
    };

    auto comparar = [](const NodoCola& a, const NodoCola& b) {
        return a.puntaje_f > b.puntaje_f;
    };

    std::priority_queue<NodoCola, std::vector<NodoCola>, decltype(comparar)> conjunto_abierto(comparar);

    std::vector<float> puntaje_g(cuadricula.size(), kInfinito);
    std::vector<float> puntaje_f(cuadricula.size(), kInfinito);
    std::vector<int> viene_de(cuadricula.size(), -1);
    std::vector<bool> cerrado(cuadricula.size(), false);
    std::vector<bool> generado(cuadricula.size(), false);

    puntaje_g[id_inicio] = 0.0f;
    puntaje_f[id_inicio] = heuristica(cuadricula[id_inicio], cuadricula[id_fin], permitir_diagonal);
    conjunto_abierto.push({id_inicio, puntaje_f[id_inicio]});
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

    while (!conjunto_abierto.empty()) {
        NodoCola actual = conjunto_abierto.top();
        conjunto_abierto.pop();

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
            const float g_tentativo = puntaje_g[actual.id] + costo_paso;

            if (g_tentativo < puntaje_g[id_vecino]) {
                viene_de[id_vecino] = actual.id;
                puntaje_g[id_vecino] = g_tentativo;
                puntaje_f[id_vecino] = g_tentativo + heuristica(cuadricula[id_vecino], cuadricula[id_fin], permitir_diagonal);
                conjunto_abierto.push({id_vecino, puntaje_f[id_vecino]});
                if (!generado[id_vecino]) {
                    generado[id_vecino] = true;
                    ejecucion.orden_generados.push_back(id_vecino);
                }
            }
        }
    }

    return ejecucion;
}

std::vector<int> encontrarRutaAEstrella(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal
) {
    return ejecutarAEstrella(cuadricula, total_columnas, total_filas, id_inicio, id_fin, permitir_diagonal).ruta;
}
