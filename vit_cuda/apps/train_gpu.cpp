#include "cuda_transformer.cuh"
#include "utils/logger.hpp"
#include "utils/mnist_loader.hpp"
#include "utils/metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct AdamGPU {
    float lr, beta1, beta2, eps, wd, grad_clip;
    int paso = 0;

    std::vector<vit::cuda::CudaTensor*> params;
    std::vector<vit::cuda::CudaTensor*> grads;
    std::vector<std::vector<float>> m, v;

    AdamGPU(std::vector<vit::cuda::CudaTensor*> p,
            std::vector<vit::cuda::CudaTensor*> g,
            float lr_ = 1e-3f, float wd_ = 1e-4f, float clip = 1.0f)
        : lr(lr_), beta1(0.9f), beta2(0.999f), eps(1e-8f),
          wd(wd_), grad_clip(clip), params(p), grads(g) {
        for (auto* pp : p) {
            m.emplace_back(pp->numel(), 0.f);
            v.emplace_back(pp->numel(), 0.f);
        }
    }

    void recortar_norma_global() {
        if (grad_clip <= 0.f) return;
        double sq = 0.0;
        for (auto* g : grads) {
            std::vector<float> h; g->to_host(h);
            for (float x : h) sq += (double)x * x;
        }
        float norma = std::sqrt((float)sq);
        if (norma > grad_clip) {
            float escala = grad_clip / (norma + 1e-6f);
            for (auto* g : grads) {
                std::vector<float> h; g->to_host(h);
                for (auto& x : h) x *= escala;
                g->from_host(h);
            }
        }
    }

    void zero_grad() { for (auto* g : grads) g->zero_(); }

    void step() {
        ++paso;
        recortar_norma_global();

        float bc1  = 1.f - std::pow(beta1, (float)paso);
        float bc2  = 1.f - std::pow(beta2, (float)paso);
        float lr_t = lr * std::sqrt(bc2) / bc1;

        for (size_t i = 0; i < params.size(); ++i) {
            std::vector<float> hp, hg;
            params[i]->to_host(hp);
            grads[i]->to_host(hg);

            for (int j = 0; j < (int)hp.size(); ++j) {
                float gj = hg[j];
                hp[j] -= lr * wd * hp[j];
                m[i][j] = beta1 * m[i][j] + (1.f - beta1) * gj;
                v[i][j] = beta2 * v[i][j] + (1.f - beta2) * gj * gj;
                hp[j] -= lr_t * m[i][j] / (std::sqrt(v[i][j]) + eps);
            }
            params[i]->from_host(hp);
        }
    }

    void programar_lr_coseno(int epoca, int total, float lr_min = 1e-5f) {
        float prog = (float)epoca / (float)total;
        float fac  = 0.5f * (1.f + std::cos(3.14159265f * prog));
        lr = lr_min + (lr - lr_min) * fac;
    }
};

static std::string arg(int argc, char** argv,
                        const std::string& clave, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == clave) return argv[i + 1];
    return def;
}

int main(int argc, char** argv) {
    auto& log = vit::utils::get_logger();

    std::string carpeta_datos = arg(argc, argv, "--datos",      "../data");
    int epocas                = std::stoi(arg(argc, argv, "--epocas",    "10"));
    int tam_lote              = std::stoi(arg(argc, argv, "--lote",      "128"));
    float lr                  = std::stof(arg(argc, argv, "--lr",        "1e-3"));
    float wd                  = std::stof(arg(argc, argv, "--wd",        "1e-4"));
    std::string carpeta_ckpt  = arg(argc, argv, "--checkpoint", "./checkpoints");
    int log_cada              = std::stoi(arg(argc, argv, "--log-cada",  "50"));

    log.separator();
    log.info("Transformador Visual (ViT) - MNIST [CUDA / GPU]");
    log.info("  carpeta datos : " + carpeta_datos);
    log.info("  epocas        : " + std::to_string(epocas));
    log.info("  tamanio lote  : " + std::to_string(tam_lote));
    log.info("  tasa aprend.  : " + std::to_string(lr));
    log.separator();

    {
        int device; cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        log.info("GPU detectada: " + std::string(prop.name));
        log.info("  VRAM: " + std::to_string(prop.totalGlobalMem / (1 << 20)) + " MB");
        log.info("  Multiprocesadores: " + std::to_string(prop.multiProcessorCount));
    }

    vit::utils::MNISTLoader cargador(carpeta_datos);
    if (!cargador.files_exist()) {
        log.error("Archivos MNIST no encontrados. Consulte el README para las instrucciones.");
        return 1;
    }

    log.info("Cargando MNIST...");
    auto entreno = cargador.load_train();
    auto prueba  = cargador.load_test();
    log.info("  entrenamiento : " + std::to_string(entreno.num_samples) + " muestras");
    log.info("  prueba        : " + std::to_string(prueba.num_samples)  + " muestras");

    vit::ViTConfig cfg;
    vit::cuda::CudaViT modelo(cfg);

    AdamGPU optimizador(modelo.parameters(), modelo.gradients(), lr, wd, 1.0f);

    std::mt19937 rng(2024);
    int n_entreno       = entreno.num_samples;
    int lotes_por_epoca = (n_entreno + tam_lote - 1) / tam_lote;

    for (int epoca = 1; epoca <= epocas; ++epoca) {
        auto t0 = std::chrono::steady_clock::now();
        vit::utils::MetricsTracker rastreador;
        rastreador.reset();

        std::vector<int> perm(n_entreno);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        optimizador.programar_lr_coseno(epoca - 1, epocas, 1e-5f);

        for (int paso = 0; paso < lotes_por_epoca; ++paso) {
            int inicio = paso * tam_lote;
            int fin    = std::min(inicio + tam_lote, n_entreno);
            int B      = fin - inicio;

            std::vector<float> x_host(B * 784);
            std::vector<int>   etiquetas(B);
            for (int i = 0; i < B; ++i) {
                int idx = perm[inicio + i];
                for (int p = 0; p < 784; ++p)
                    x_host[i * 784 + p] = entreno.images[idx][p];
                etiquetas[i] = entreno.labels[idx];
            }

            optimizador.zero_grad();
            vit::cuda::CudaTensor logits = modelo.forward(x_host, B);
            auto [perdida, correctos]    = modelo.loss_and_accuracy(logits, etiquetas, B);

            {
                int C = cfg.num_classes;
                std::vector<float> h_logits(B * C), h_grad(B * C);
                logits.to_host(h_logits);

                for (int b = 0; b < B; ++b) {
                    float mx = *std::max_element(&h_logits[b*C], &h_logits[b*C]+C);
                    float suma = 0.f;
                    for (int c = 0; c < C; ++c) {
                        h_grad[b*C+c] = std::exp(h_logits[b*C+c] - mx);
                        suma += h_grad[b*C+c];
                    }
                    for (int c = 0; c < C; ++c) {
                        h_grad[b*C+c] /= suma * B;
                        if (c == etiquetas[b]) h_grad[b*C+c] -= 1.f / B;
                    }
                }

                vit::cuda::CudaTensor grad_logits({B, C});
                grad_logits.from_host(h_grad);
                modelo.backward(grad_logits, B);
            }

            optimizador.step();
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
            std::string ruta = carpeta_ckpt + "/vit_gpu_epoca" +
                               std::to_string(epoca) + ".bin";
            try {
                modelo.save(ruta);
                log.info("  checkpoint guardado: " + ruta);
            } catch (const std::exception& e) {
                log.warn("  fallo checkpoint: " + std::string(e.what()));
            }
        }
    }

    log.separator();
    log.info("Evaluando en conjunto de prueba...");
    {
        vit::utils::MetricsTracker rastreador; rastreador.reset();
        int n = prueba.num_samples;
        for (int inicio = 0; inicio < n; inicio += tam_lote) {
            int fin = std::min(inicio + tam_lote, n);
            int B   = fin - inicio;
            std::vector<float> x_host(B * 784);
            std::vector<int>   etiquetas(B);
            for (int i = 0; i < B; ++i) {
                for (int p = 0; p < 784; ++p)
                    x_host[i * 784 + p] = prueba.images[inicio + i][p];
                etiquetas[i] = prueba.labels[inicio + i];
            }
            auto logits = modelo.forward(x_host, B);
            auto [perdida, correctos] = modelo.loss_and_accuracy(logits, etiquetas, B);
            rastreador.update(perdida, correctos, B);
        }
        auto stats = rastreador.compute(0.0);
        log.log_test(stats.avg_loss, stats.accuracy);
    }
    log.separator();

    return 0;
}
