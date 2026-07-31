#include "vit/transformer.hpp"
#include "vit/layers.hpp"
#include "utils/logger.hpp"
#include "utils/mnist_loader.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static std::string arg(int argc, char** argv,
                        const std::string& clave, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == clave) return argv[i + 1];
    return def;
}

int main(int argc, char** argv) {
    auto& log = vit::utils::get_logger();

    std::string ckpt     = arg(argc, argv, "--checkpoint", "");
    std::string datos    = arg(argc, argv, "--datos",      "../data");
    int         tam_lote = std::stoi(arg(argc, argv, "--lote", "256"));

    if (ckpt.empty()) {
        log.error("Uso: vit_infer --checkpoint <ruta> [--datos <carpeta>]");
        return 1;
    }

    log.separator();
    log.info("ViT - Inferencia");
    log.info("  checkpoint : " + ckpt);
    log.info("  datos      : " + datos);
    log.separator();

    vit::ViTConfig cfg;
    vit::ViT modelo(cfg);

    try {
        modelo.load(ckpt);
        log.info("Checkpoint cargado exitosamente.");
    } catch (const std::exception& e) {
        log.error("Error al cargar checkpoint: " + std::string(e.what()));
        return 1;
    }
    modelo.set_training(false);

    vit::utils::MNISTLoader cargador(datos);
    if (!cargador.files_exist()) {
        log.error("No se encontraron los datos en: " + datos);
        return 1;
    }
    auto prueba = cargador.load_test();
    log.info("Muestras de prueba: " + std::to_string(prueba.num_samples));

    int conf[10][10] = {};
    int total_correctos = 0;
    int n = prueba.num_samples;

    for (int inicio = 0; inicio < n; inicio += tam_lote) {
        int fin = std::min(inicio + tam_lote, n);
        int B   = fin - inicio;

        vit::Tensor x({B, 784});
        std::vector<int> etiquetas(B);
        for (int i = 0; i < B; ++i) {
            for (int p = 0; p < 784; ++p)
                x.at(i, p) = prueba.images[inicio + i][p];
            etiquetas[i] = prueba.labels[inicio + i];
        }

        vit::Tensor logits = modelo.forward(x);

        for (int b = 0; b < B; ++b) {
            int real = etiquetas[b];
            int pred = 0;
            float mx = logits.at(b, 0);
            for (int c = 1; c < 10; ++c)
                if (logits.at(b, c) > mx) { mx = logits.at(b, c); pred = c; }
            conf[real][pred]++;
            if (real == pred) ++total_correctos;
        }
    }

    float exactitud_global = 100.f * total_correctos / n;

    log.separator();
    log.info("Exactitud global: " + std::to_string(exactitud_global) + "%");
    log.separator();
    log.info("Exactitud por digito:");

    for (int c = 0; c < 10; ++c) {
        int total_c = 0;
        for (int p = 0; p < 10; ++p) total_c += conf[c][p];
        float acc = total_c > 0 ? 100.f * conf[c][c] / total_c : 0.f;
        std::ostringstream oss;
        oss << "  Digito " << c << ": " << std::fixed << std::setprecision(1)
            << acc << "%  (" << conf[c][c] << "/" << total_c << ")";
        log.info(oss.str());
    }

    log.separator();
    log.info("Matriz de confusion (filas=real, columnas=predicho):");
    std::cout << "\n      ";
    for (int c = 0; c < 10; ++c) std::cout << std::setw(5) << c;
    std::cout << "\n";
    for (int real = 0; real < 10; ++real) {
        std::cout << "  " << real << "  |";
        for (int pred = 0; pred < 10; ++pred)
            std::cout << std::setw(5) << conf[real][pred];
        std::cout << "\n";
    }
    std::cout << "\n";

    return 0;
}
