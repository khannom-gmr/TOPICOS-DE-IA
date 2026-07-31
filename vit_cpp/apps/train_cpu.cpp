#include "vit/transformer.hpp"
#include "vit/layers.hpp"
#include "vit/optimizer.hpp"
#include "utils/logger.hpp"
#include "utils/mnist_loader.hpp"
#include "utils/metrics.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static std::string arg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == key) return argv[i + 1];
    return def;
}

static std::vector<int> permutacion(int n, std::mt19937& rng) {
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}

static vit::utils::EpochStats evaluar(vit::ViT& modelo,
                                       const vit::utils::MNISTDataset& datos,
                                       int tam_lote) {
    modelo.set_training(false);
    vit::utils::MetricsTracker rastreador;
    rastreador.reset();

    for (int inicio = 0; inicio < datos.num_samples; inicio += tam_lote) {
        int fin = std::min(inicio + tam_lote, datos.num_samples);
        int B   = fin - inicio;

        vit::Tensor x({B, 784});
        std::vector<int> etiquetas(B);
        for (int i = 0; i < B; ++i) {
            for (int p = 0; p < 784; ++p)
                x.at(i, p) = datos.images[inicio + i][p];
            etiquetas[i] = datos.labels[inicio + i];
        }

        vit::Tensor logits = modelo.forward(x);
        auto [perdida, grad] = vit::cross_entropy(logits, etiquetas);
        int correctos = vit::utils::count_correct(logits.data, etiquetas, B, 10);
        rastreador.update(perdida, correctos, B);
    }

    return rastreador.compute(0.0);
}

int main(int argc, char** argv) {
    auto& log = vit::utils::get_logger();

    std::string carpeta_datos = arg(argc, argv, "--datos",      "../data");
    int epocas                = std::stoi(arg(argc, argv, "--epocas",    "10"));
    int tam_lote              = std::stoi(arg(argc, argv, "--lote",      "64"));
    float lr                  = std::stof(arg(argc, argv, "--lr",        "1e-3"));
    float wd                  = std::stof(arg(argc, argv, "--wd",        "1e-4"));
    std::string carpeta_ckpt  = arg(argc, argv, "--checkpoint", "./checkpoints");
    int log_cada              = std::stoi(arg(argc, argv, "--log-cada",  "100"));

    log.separator();
    log.info("Transformador Visual (ViT) - MNIST [CPU]");
    log.info("  carpeta datos : " + carpeta_datos);
    log.info("  epocas        : " + std::to_string(epocas));
    log.info("  tamanio lote  : " + std::to_string(tam_lote));
    log.info("  tasa aprend.  : " + std::to_string(lr));
    log.separator();

    vit::utils::MNISTLoader cargador(carpeta_datos);
    if (!cargador.files_exist()) {
        log.error("No se encontraron los archivos MNIST en: " + carpeta_datos);
        log.error("Consulte el README para las instrucciones de configuracion de datos.");
        return 1;
    }

    log.info("Cargando MNIST...");
    auto entreno = cargador.load_train();
    auto prueba  = cargador.load_test();
    log.info("  entrenamiento : " + std::to_string(entreno.num_samples) + " muestras");
    log.info("  prueba        : " + std::to_string(prueba.num_samples)  + " muestras");

    vit::ViTConfig cfg;
    log.info("Configuracion ViT:");
    log.info("  parches     : " + std::to_string(cfg.num_patches()) +
             " x " + std::to_string(cfg.patch_dim()) + "d");
    log.info("  dim embed   : " + std::to_string(cfg.embed_dim));
    log.info("  cabezas     : " + std::to_string(cfg.num_heads));
    log.info("  capas       : " + std::to_string(cfg.num_layers));
    log.info("  dim MLP     : " + std::to_string(cfg.mlp_dim));
    log.separator();

    vit::ViT modelo(cfg);

    vit::AdamOptimizer::Config cfg_opt;
    cfg_opt.lr = lr; cfg_opt.weight_decay = wd; cfg_opt.grad_clip = 1.0f;
    vit::AdamOptimizer optimizador(modelo.parameters(), modelo.gradients(), cfg_opt);

    std::mt19937 rng(2024);
    int n_entreno          = entreno.num_samples;
    int lotes_por_epoca    = (n_entreno + tam_lote - 1) / tam_lote;

    modelo.set_training(true);
    for (int epoca = 1; epoca <= epocas; ++epoca) {
        auto t0 = std::chrono::steady_clock::now();
        vit::utils::MetricsTracker rastreador;
        rastreador.reset();

        auto perm = permutacion(n_entreno, rng);
        optimizador.cosine_schedule(epoca - 1, epocas, 1e-5f);

        for (int paso = 0; paso < lotes_por_epoca; ++paso) {
            int inicio = paso * tam_lote;
            int fin    = std::min(inicio + tam_lote, n_entreno);
            int B      = fin - inicio;

            vit::Tensor x({B, 784});
            std::vector<int> etiquetas(B);
            for (int i = 0; i < B; ++i) {
                int idx = perm[inicio + i];
                for (int p = 0; p < 784; ++p)
                    x.at(i, p) = entreno.images[idx][p];
                etiquetas[i] = entreno.labels[idx];
            }

            optimizador.zero_grad();
            vit::Tensor logits = modelo.forward(x);
            auto [perdida, grad] = vit::cross_entropy(logits, etiquetas);
            modelo.backward(grad);
            optimizador.step();

            int correctos = vit::utils::count_correct(logits.data, etiquetas, B, 10);
            rastreador.update(perdida, correctos, B);

            if ((paso + 1) % log_cada == 0 || paso == lotes_por_epoca - 1)
                log.log_batch(paso + 1, lotes_por_epoca,
                              rastreador.running_loss(),
                              rastreador.running_accuracy());
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        auto stats = rastreador.compute(elapsed);
        log.log_epoch(epoca, epocas, stats.avg_loss, stats.accuracy, elapsed);

        if (epoca % 5 == 0) {
            std::string ruta = carpeta_ckpt + "/vit_cpu_epoca" +
                               std::to_string(epoca) + ".bin";
            try {
                modelo.save(ruta);
                log.info("  checkpoint guardado: " + ruta);
            } catch (const std::exception& e) {
                log.warn("  fallo al guardar checkpoint: " + std::string(e.what()));
            }
        }
    }

    log.separator();
    log.info("Evaluando en conjunto de prueba...");
    auto stats_prueba = evaluar(modelo, prueba, tam_lote);
    log.log_test(stats_prueba.avg_loss, stats_prueba.accuracy);
    log.separator();

    return 0;
}
