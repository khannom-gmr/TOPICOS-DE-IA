#include "dijkstra_algorithm.h"
#include <queue>
#include <limits>
#include <cmath>
#include <algorithm>

namespace {
constexpr float INF = std::numeric_limits<float>::infinity();

int aId(int fila, int col, int total_columnas) {
    return fila * total_columnas + col;
}

bool dentro(int f, int c, int filas, int cols) {
    return f >= 0 && f < filas && c >= 0 && c < cols;
}

std::vector<int> reconstruirRuta(const std::vector<int>& parent, int actual) {
    std::vector<int> ruta;
    while (actual != -1) {
        ruta.push_back(actual);
        actual = parent[actual];
    }
    std::reverse(ruta.begin(), ruta.end());
    return ruta;
}
}

EjecucionAEstrella ejecutarDijkstra(
    const std::vector<NodoCuadricula>& cuadricula,
    int total_columnas,
    int total_filas,
    int id_inicio,
    int id_fin,
    bool permitir_diagonal
) {
    EjecucionAEstrella ejecucion;

    std::vector<float> costo(cuadricula.size(), INF);
    std::vector<int> parent(cuadricula.size(), -1);
    std::vector<bool> cerrado(cuadricula.size(), false);
    std::vector<bool> en_frontier(cuadricula.size(), false);

    struct Nodo {
        int id;
        float costo;
    };

    auto cmp = [](Nodo a, Nodo b) {
        return a.costo > b.costo;
    };

    std::priority_queue<Nodo, std::vector<Nodo>, decltype(cmp)> pq(cmp);

    costo[id_inicio] = 0.0f;
    pq.push({id_inicio, 0.0f});
    en_frontier[id_inicio] = true;
    ejecucion.orden_generados.push_back(id_inicio);

    const int dirs[8][2] = {
        {0,1},{1,0},{0,-1},{-1,0},
        {1,1},{1,-1},{-1,1},{-1,-1}
    };

    while (!pq.empty()) {
        Nodo actual = pq.top(); pq.pop();

        if (cerrado[actual.id]) continue;

        cerrado[actual.id] = true;
        ejecucion.orden_expandidos.push_back(actual.id);

        if (actual.id == id_fin) {
            ejecucion.ruta = reconstruirRuta(parent, id_fin);
            return ejecucion;
        }

        int f = cuadricula[actual.id].row;
        int c = cuadricula[actual.id].col;

        for (int i = 0; i < (permitir_diagonal ? 8 : 4); i++) {

            int nf = f + dirs[i][0];
            int nc = c + dirs[i][1];

            if (!dentro(nf, nc, total_filas, total_columnas)) continue;

            int vecino = aId(nf, nc, total_columnas);
            if (cuadricula[vecino].bloqueado) continue;

            float costo_paso = (i >= 4) ? std::sqrt(2.0f) : 1.0f;
            float nuevo_costo = costo[actual.id] + costo_paso;

            if (!en_frontier[vecino]) {
                costo[vecino] = nuevo_costo;
                parent[vecino] = actual.id;
                pq.push({vecino, nuevo_costo});

                en_frontier[vecino] = true;
                ejecucion.orden_generados.push_back(vecino);
            }
            else if (nuevo_costo < costo[vecino] && !cerrado[vecino]) {
                costo[vecino] = nuevo_costo;
                parent[vecino] = actual.id;

                pq.push({vecino, nuevo_costo});
            }
        }
    }

    return ejecucion;
}
