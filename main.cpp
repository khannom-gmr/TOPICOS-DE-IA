#include <iostream>
#include <vector>
using namespace std;

// FUNCIÓN DE ACTIVACIÓN
int funcion_activacion(float entrada_neta) {
    if (entrada_neta >= 0)
        return 1;
    else
        return 0;
}

// REGLA DE PROPAGACIÓN
float calcular_entrada_neta(vector<float> pesos, vector<int> entradas) {
    float suma = 0;

    for (int i = 0; i < pesos.size(); i++) {
        suma += pesos[i] * entradas[i];
    }

    return suma;
}

// ESTADO DE ACTIVACIÓN
int estado_activacion(vector<int> entradas, vector<float> pesos) {
    float z = calcular_entrada_neta(pesos, entradas);
    return funcion_activacion(z);
}

// ENTRENAMIENTO DEL PERCEPTRÓN
vector<float> entrenar_perceptron(
    vector<vector<int>> datos_entrada,
    vector<int> salidas_deseadas,
    float tasa_aprendizaje = 0.5,
    int epocas = 100
) {

    int num_caracteristicas = datos_entrada[0].size();

    // Inicializar pesos en 0
    vector<float> pesos(num_caracteristicas, 0);

    for (int epoca = 0; epoca < epocas; epoca++) {

        cout << "\nEpoca " << epoca + 1 << endl;

        int errores = 0;

        for (int i = 0; i < datos_entrada.size(); i++) {

            vector<int> x_i = datos_entrada[i];
            int t = salidas_deseadas[i];

            float z = calcular_entrada_neta(pesos, x_i);

            int y = funcion_activacion(z);

            int error = t - y;

            // MOSTRAR DATOS
            cout << "Entrada x: ";

            for (int j = 0; j < x_i.size(); j++) {
                cout << x_i[j] << " ";
            }

            cout << " z: " << z
                 << " y: " << y
                 << " t: " << t
                 << " error: " << error
                 << " pesos: ";

            for (int j = 0; j < pesos.size(); j++) {
                cout << pesos[j] << " ";
            }

            cout << endl;

            // ACTUALIZAR PESOS
            if (error != 0) {

                for (int j = 0; j < pesos.size(); j++) {
                    pesos[j] = pesos[j] + tasa_aprendizaje * error * x_i[j];
                }

                errores++;
            }
        }

        // SI YA NO HAY ERRORES
        if (errores == 0) {
            cout << "\nEntrenamiento finalizado." << endl;
            break;
        }
    }

    return pesos;
}

int main() {

    // ENTRADAS (x0 = 1)
    vector<vector<int>> datos_entrada = {
        {1, 0, 0},
        {1, 1, 0},
        {1, 0, 1},
        {1, 1, 1}
    };

    // SALIDAS DESEADAS PARA AND
    vector<int> salidas_deseadas = {0, 0, 0, 1};

    // ENTRENAMIENTO
    vector<float> pesos_finales =
        entrenar_perceptron(datos_entrada, salidas_deseadas);

    // MOSTRAR PESOS FINALES
    cout << "\nPesos finales aprendidos: ";

    for (int i = 0; i < pesos_finales.size(); i++) {
        cout << pesos_finales[i] << " ";
    }

    // PRUEBAS
    cout << "\n\nPrueba del perceptron entrenado:" << endl;

    for (int i = 0; i < datos_entrada.size(); i++) {

        int y = estado_activacion(datos_entrada[i], pesos_finales);

        cout << "Entrada X: ";

        for (int j = 0; j < datos_entrada[i].size(); j++) {
            cout << datos_entrada[i][j] << " ";
        }

        cout << " -> Salida Y: " << y << endl;
    }

    return 0;
}
