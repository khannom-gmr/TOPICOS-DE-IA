#include "dijkstra_algorithm.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "asterisco.h"
#include "astar_algorithm.h"

using namespace std;

const int ANCHO_VENTANA = 900;
const int ALTO_VENTANA = 700;
const int ESPACIADO_CUADRICULA = 60;
bool usar_dijkstra = false;

struct NodoVisual {
    int id;
    int columna;
    int fila;
    float x;
    float y;
    bool eliminado = false;
    bool generado = false;
    bool en_ruta = false;
    bool explorado = false;
};

int total_columnas;
int total_filas;
vector<NodoVisual> nodos;
vector<pair<int, int>> aristas_ruta;
int id_inicio = -1;
int id_fin = -1;
vector<int> orden_generados;
vector<int> orden_expandidos;
size_t progreso_expandidos = 0;
double tiempo_ultimo_paso = 0.0;
double intervalo_paso = 0.05;
bool animacion_activa = false;
vector<int> ruta_pendiente;

void construirAristasRuta(const vector<int>& ruta);
void actualizarAnimacion(double ahora);
void refrescarRuta();
int seleccionarNodoEn(double x, double y);
void callbackBotonMouse(GLFWwindow* ventana, int boton, int accion, int mods);

void construirCuadricula() {
    nodos.clear();
    aristas_ruta.clear();

    float margen = 40.0f;
    total_columnas = (ANCHO_VENTANA - static_cast<int>(margen * 2)) / ESPACIADO_CUADRICULA + 1;
    total_filas = (ALTO_VENTANA - static_cast<int>(margen * 2)) / ESPACIADO_CUADRICULA + 1;

    for (int fila = 0; fila < total_filas; fila++) {
        for (int columna = 0; columna < total_columnas; columna++) {
            NodoVisual nodo;
            nodo.id = fila * total_columnas + columna;
            nodo.columna = columna;
            nodo.fila = fila;
            nodo.x = margen + columna * ESPACIADO_CUADRICULA;
            nodo.y = margen + fila * ESPACIADO_CUADRICULA;
            nodos.push_back(nodo);
        }
    }
}

vector<int> obtenerVecinos(int id) {
    int fila = nodos[id].fila;
    int columna = nodos[id].columna;
    vector<int> vecinos;

    auto intentarAgregar = [&](int df, int dc) {
        int nueva_fila = fila + df;
        int nueva_columna = columna + dc;
        if (nueva_fila >= 0 && nueva_fila < total_filas && nueva_columna >= 0 && nueva_columna < total_columnas) {
            int id_vecino = nueva_fila * total_columnas + nueva_columna;
            if (!nodos[id_vecino].eliminado) {
                vecinos.push_back(id_vecino);
            }
        }
    };

    intentarAgregar(0, 1);
    intentarAgregar(1, 0);
    intentarAgregar(0, -1);
    intentarAgregar(-1, 0);
    intentarAgregar(1, 1);
    intentarAgregar(1, -1);
    intentarAgregar(-1, 1);
    intentarAgregar(-1, -1);

    return vecinos;
}

void eliminarAleatoriosPorCantidad(int cantidad) {
    vector<int> candidatos;
    candidatos.reserve(nodos.size());

    for (const auto& nodo : nodos) {
        if (nodo.id != id_inicio && nodo.id != id_fin) {
            candidatos.push_back(nodo.id);
        }
    }

    random_device rd;
    mt19937 generador(rd());
    shuffle(candidatos.begin(), candidatos.end(), generador);

    cantidad = max(0, min(cantidad, static_cast<int>(candidatos.size())));

    for (auto& nodo : nodos) {
        nodo.eliminado = false;
    }

    for (int i = 0; i < cantidad; i++) {
        nodos[candidatos[i]].eliminado = true;
    }
}

void pedirYEliminarAleatorios() {
    int maximo = 0;
    for (const auto& nodo : nodos) {
        if (nodo.id != id_inicio && nodo.id != id_fin) {
            maximo++;
        }
    }

    cout << "Cuantos nodos quieres borrar (0-" << maximo << "): ";
    int cantidad = 0;
    if (!(cin >> cantidad)) {
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cout << "Entrada invalida" << endl;
        return;
    }

    if (cantidad < 0) {
        cantidad = 0;
    }
    if (cantidad > maximo) {
        cantidad = maximo;
    }

    eliminarAleatoriosPorCantidad(cantidad);
    refrescarRuta();
}

EjecucionAEstrella ejecutarBusquedaAEstrella(int origen, int destino) {
    vector<NodoCuadricula> cuadricula;
    cuadricula.reserve(nodos.size());

    for (const auto& nodo : nodos) {
        cuadricula.push_back({nodo.id, nodo.columna, nodo.fila, nodo.eliminado});
    }

    return ejecutarAEstrella(cuadricula, total_columnas, total_filas, origen, destino, true);
}

EjecucionAEstrella ejecutarBusquedaDijkstra(int origen, int destino) {
    vector<NodoCuadricula> cuadricula;

    for (const auto& nodo : nodos) {
        cuadricula.push_back({nodo.id, nodo.columna, nodo.fila, nodo.eliminado});
    }

    return ejecutarDijkstra(cuadricula, total_columnas, total_filas, origen, destino, true);
}


void reiniciarEstadoRuta() {
    aristas_ruta.clear();
    orden_generados.clear();
    orden_expandidos.clear();
    ruta_pendiente.clear();
    progreso_expandidos = 0;
    animacion_activa = false;

    for (auto& nodo : nodos) {
        nodo.generado = false;
        nodo.en_ruta = false;
        nodo.explorado = false;
    }
}

void construirAristasRuta(const vector<int>& ruta) {
    aristas_ruta.clear();

    for (int id : ruta) {
        nodos[id].en_ruta = true;
    }

    for (size_t i = 1; i < ruta.size(); i++) {
        aristas_ruta.push_back({ruta[i - 1], ruta[i]});
    }
}

void refrescarRuta() {
    reiniciarEstadoRuta();
    if (id_inicio < 0 || id_fin < 0) {
        return;
    }

    EjecucionAEstrella ejecucion;

    if (usar_dijkstra) {
        cout << ">>> Ejecutando DIJKSTRA\n";
        ejecucion = ejecutarBusquedaDijkstra(id_inicio, id_fin);
    }
    else {
        cout << ">>> Ejecutando A*\n";
        ejecucion = ejecutarBusquedaAEstrella(id_inicio, id_fin);
    }
    orden_generados = ejecucion.orden_generados;
    orden_expandidos = ejecucion.orden_expandidos;
    ruta_pendiente = ejecucion.ruta;
    tiempo_ultimo_paso = glfwGetTime();

    for (int id : orden_generados) {
        if (id >= 0 && id < static_cast<int>(nodos.size())) {
            nodos[id].generado = true;
        }
    }

    if (orden_expandidos.empty()) {
        if (!ruta_pendiente.empty()) {
            construirAristasRuta(ruta_pendiente);
        }
        return;
    }

    animacion_activa = true;
}

void actualizarAnimacion(double ahora) {
    if (!animacion_activa) {
        return;
    }

    if (ahora - tiempo_ultimo_paso < intervalo_paso) {
        return;
    }

    tiempo_ultimo_paso = ahora;

    if (progreso_expandidos < orden_expandidos.size()) {
        int id = orden_expandidos[progreso_expandidos];
        if (id >= 0 && id < static_cast<int>(nodos.size())) {
            nodos[id].explorado = true;
        }
        progreso_expandidos++;
    }

    if (progreso_expandidos >= orden_expandidos.size()) {
        animacion_activa = false;
        if (!ruta_pendiente.empty()) {
            construirAristasRuta(ruta_pendiente);
        }
    }
}

int seleccionarNodoEn(double x, double y) {
    const float radio_maximo = static_cast<float>(ESPACIADO_CUADRICULA) * 0.40f;
    const float distancia_maxima2 = radio_maximo * radio_maximo;

    int mejor_id = -1;
    float mejor_distancia2 = distancia_maxima2;

    for (const auto& nodo : nodos) {
        const float dx = static_cast<float>(x) - nodo.x;
        const float dy = static_cast<float>(y) - nodo.y;
        const float distancia2 = dx * dx + dy * dy;
        if (distancia2 <= mejor_distancia2) {
            mejor_distancia2 = distancia2;
            mejor_id = nodo.id;
        }
    }

    return mejor_id;
}

void callbackBotonMouse(GLFWwindow* ventana, int boton, int accion, int) {
    if (accion != GLFW_PRESS) {
        return;
    }

    if (boton != GLFW_MOUSE_BUTTON_LEFT && boton != GLFW_MOUSE_BUTTON_RIGHT) {
        return;
    }

    double cursor_x;
    double cursor_y;
    glfwGetCursorPos(ventana, &cursor_x, &cursor_y);

    int ancho_actual;
    int alto_actual;
    glfwGetWindowSize(ventana, &ancho_actual, &alto_actual);
    if (ancho_actual <= 0 || alto_actual <= 0) {
        return;
    }

    const double escala_x = static_cast<double>(ANCHO_VENTANA) / static_cast<double>(ancho_actual);
    const double escala_y = static_cast<double>(ALTO_VENTANA) / static_cast<double>(alto_actual);
    const double mundo_x = cursor_x * escala_x;
    const double mundo_y = cursor_y * escala_y;

    int id_seleccionado = seleccionarNodoEn(mundo_x, mundo_y);
    if (id_seleccionado < 0) {
        return;
    }

    nodos[id_seleccionado].eliminado = false;

    nodos[id_seleccionado].eliminado = !nodos[id_seleccionado].eliminado;


    if (id_seleccionado == id_inicio || id_seleccionado == id_fin) {
    nodos[id_seleccionado].eliminado = false;
    }


    refrescarRuta();
}

void reiniciarCuadriculaYExtremos() {
    construirCuadricula();
    id_inicio = 0;
    id_fin = nodos.size() - 1;
    refrescarRuta();
}

void dibujarEscena() {
    glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, ANCHO_VENTANA, ALTO_VENTANA, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glLineWidth(0.8f);
    glBegin(GL_LINES);
    for (auto& nodo : nodos) {
        if (nodo.eliminado) {
            continue;
        }
        for (int id_vecino : obtenerVecinos(nodo.id)) {
            if (id_vecino > nodo.id) {
                glColor4f(0.4f, 0.4f, 0.5f, 0.5f);
                glVertex2f(nodo.x, nodo.y);
                glVertex2f(nodos[id_vecino].x, nodos[id_vecino].y);
            }
        }
    }
    glEnd();

    glLineWidth(2.5f);
    glBegin(GL_LINES);
    glColor3f(0.98f, 0.80f, 0.26f);
    for (auto& arista : aristas_ruta) {
        glVertex2f(nodos[arista.first].x, nodos[arista.first].y);
        glVertex2f(nodos[arista.second].x, nodos[arista.second].y);
    }
    glEnd();

    for (auto& nodo : nodos) {
        float radio = 4.5f;
        int segmentos = 16;

        if (nodo.id == id_inicio) {
            glColor3f(0.94f, 0.31f, 0.22f);
            radio = 6.0f;
        } else if (nodo.id == id_fin) {
            glColor3f(0.18f, 0.75f, 0.85f);
            radio = 6.0f;
        } else if (nodo.eliminado) {
            glColor4f(0.75f, 0.22f, 0.18f, 0.5f);
            radio = 3.0f;
        } else if (nodo.en_ruta) {
            glColor3f(0.11f, 0.62f, 0.46f);
            radio = 5.0f;
        } else if (nodo.explorado) {
            glColor3f(0.95f, 0.67f, 0.17f);
            radio = 5.0f;
        } else if (nodo.generado) {
            glColor3f(0.61f, 0.48f, 0.94f);
            radio = 4.8f;
        } else {
            glColor3f(0.22f, 0.54f, 0.87f);
            radio = 4.5f;
        }

        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(nodo.x, nodo.y);
        for (int i = 0; i <= segmentos; i++) {
            float angulo = i * 2.0f * 3.14159f / segmentos;
            glVertex2f(nodo.x + radio * cosf(angulo), nodo.y + radio * sinf(angulo));
        }
        glEnd();
    }

    glLineWidth(1.0f);
}

void callbackTeclado(GLFWwindow* ventana, int tecla, int, int accion, int) {

    if (accion != GLFW_PRESS) return;

    if (tecla == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(ventana, true);
    }

    else if (tecla == GLFW_KEY_R) {
        reiniciarCuadriculaYExtremos();
    }

    else if (tecla == GLFW_KEY_O) {
        pedirYEliminarAleatorios();
    }

    else if (tecla == GLFW_KEY_C) {
        for (auto& nodo : nodos) {
            nodo.eliminado = false;
        }
        refrescarRuta();
    }

    else if (tecla == GLFW_KEY_SPACE) {
        refrescarRuta();
    }


    else if (tecla == GLFW_KEY_D) {
        usar_dijkstra = !usar_dijkstra;

        if (usar_dijkstra) {
            cout << "Modo: DIJKSTRA\n";
        } else {
            cout << "Modo: A*\n";
        }

        refrescarRuta();
    }
}


int runAsteriscoApp() {
    if (!glfwInit()) {
        cout << "Error starting GLFW" << endl;
        return -1;
    }

    GLFWwindow* ventana = glfwCreateWindow(ANCHO_VENTANA, ALTO_VENTANA, "Grid Graph - A*", nullptr, nullptr);
    if (!ventana) {
        cout << "Error creating window" << endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(ventana);
    glfwSetKeyCallback(ventana, callbackTeclado);
    glfwSetMouseButtonCallback(ventana, callbackBotonMouse);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);

    cout << "Controles:" << endl;
    cout << "  R: Reiniciar malla" << endl;
    cout << "  O: Borrar nodos aleatorios por cantidad" << endl;
    cout << "  C: Limpiar obstaculos" << endl;
    cout << "  D: Cambiar A* / Dijkstra" << endl;
    cout << "  Espacio: Recalcular ruta A*" << endl;
    cout << "  Click izquierdo: Elegir nodo inicio" << endl;
    cout << "  Click derecho: Elegir nodo destino" << endl;
    cout << "  Morado: espacio de estados generado" << endl;
    cout << "  ESC: Salir" << endl;

    reiniciarCuadriculaYExtremos();

    while (!glfwWindowShouldClose(ventana)) {
        actualizarAnimacion(glfwGetTime());
        dibujarEscena();
        glfwSwapBuffers(ventana);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
