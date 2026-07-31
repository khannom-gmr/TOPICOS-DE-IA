#include "utils/mnist_loader.hpp"
#include <cassert>
#include <iostream>
#include <string>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; ++g_fail; } \
         else { std::cout << "[PASS] " << msg << "\n"; ++g_pass; } } while(0)

using namespace vit::utils;

int main(int argc, char** argv) {
    std::string data_dir = (argc > 1) ? argv[1] : "../data";
    std::cout << "=== test_mnist_loader (data: " << data_dir << ") ===\n";

    MNISTLoader loader(data_dir);

    if (!loader.files_exist()) {
        std::cerr << "[SKIP] Archivos MNIST no encontrados en " << data_dir
                  << ". Consulte el README para mas informacion.\n";
        return 0;
    }

    auto train = loader.load_train();
    CHECK(train.num_samples == 60000,        "train tiene 60000 muestras");
    CHECK((int)train.images.size() == 60000, "tamanio vector imagenes de entrenamiento");
    CHECK((int)train.labels.size() == 60000, "tamanio vector etiquetas de entrenamiento");
    CHECK((int)train.images[0].size() == 784, "imagen tiene 784 pixeles");

    bool all_normalized = true;
    for (float v : train.images[0])
        if (v < 0.f || v > 1.f) { all_normalized = false; break; }
    CHECK(all_normalized, "pixeles normalizados en [0,1]");

    bool labels_ok = true;
    for (int l : train.labels)
        if (l < 0 || l > 9) { labels_ok = false; break; }
    CHECK(labels_ok, "etiquetas en rango [0,9]");

    auto test = loader.load_test();
    CHECK(test.num_samples == 10000, "test tiene 10000 muestras");
    CHECK(test.labels[0] == 7, "primera etiqueta de test es 7");

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed.\n";
    return g_fail > 0 ? 1 : 0;
}
