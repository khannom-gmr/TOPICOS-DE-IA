#include <stdio.h>
#include <string.h>

#define ENTRADAS 3
#define MUESTRAS 4
#define EPOCAS   100
#define TASA     0.5

int activacion(double z) {
    return z >= 0 ? 1 : 0;
}

double entrada_neta(double pesos[], double x[]) {
    double suma = 0.0;
    for (int i = 0; i < ENTRADAS; i++)
        suma += pesos[i] * x[i];
    return suma;
}

void entrenar(double datos[][ENTRADAS], int salidas[], double pesos[]) {
    memset(pesos, 0, sizeof(double) * ENTRADAS);

    for (int epoca = 0; epoca < EPOCAS; epoca++) {
        printf("\nEpoca %d\n", epoca + 1);
        int errores = 0;

        for (int i = 0; i < MUESTRAS; i++) {
            double z = entrada_neta(pesos, datos[i]);
            int y    = activacion(z);
            int err  = salidas[i] - y;

            printf("  x=[%.0f %.0f %.0f]  z=%.2f  y=%d  t=%d  error=%d  w=[%.2f %.2f %.2f]\n",
                   datos[i][0], datos[i][1], datos[i][2],
                   z, y, salidas[i], err,
                   pesos[0], pesos[1], pesos[2]);

            if (err != 0) {
                for (int j = 0; j < ENTRADAS; j++)
                    pesos[j] += TASA * err * datos[i][j];
                errores++;
            }
        }

        if (errores == 0) {
            printf("\nEntrenamiento finalizado.\n");
            break;
        }
    }
}

void probar(double datos[][ENTRADAS], double pesos[], const char *nombre) {
    printf("\nPrueba %s:\n", nombre);
    for (int i = 0; i < MUESTRAS; i++) {
        double z = entrada_neta(pesos, datos[i]);
        int y    = activacion(z);
        printf("  x=[%.0f %.0f %.0f]  ->  y=%d\n",
               datos[i][0], datos[i][1], datos[i][2], y);
    }
}

int main() {
    double datos[MUESTRAS][ENTRADAS] = {
        {1, 0, 0},
        {1, 1, 0},
        {1, 0, 1},
        {1, 1, 1}
    };

    double pesos[ENTRADAS];

    /* AND */
    int and[] = {0, 0, 0, 1};
    printf("========== AND ==========");
    entrenar(datos, and, pesos);
    printf("\nPesos finales: [%.2f %.2f %.2f]\n", pesos[0], pesos[1], pesos[2]);
    probar(datos, pesos, "AND");

    /* OR */
    int or[] = {0, 1, 1, 1};
    printf("\n========== OR ==========");
    entrenar(datos, or, pesos);
    printf("\nPesos finales: [%.2f %.2f %.2f]\n", pesos[0], pesos[1], pesos[2]);
    probar(datos, pesos, "OR");

    return 0;
}
