#include "utils/mnist_loader.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace vit::utils {

namespace fs = std::filesystem;

MNISTLoader::MNISTLoader(const std::string& data_dir)
    : data_dir_(data_dir) {}

bool MNISTLoader::files_exist() const {
    const std::vector<std::string> required = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    };
    for (auto& f : required) {
        if (!fs::exists(data_dir_ + "/" + f)) return false;
    }
    return true;
}

MNISTDataset MNISTLoader::load_train() {
    return load(data_dir_ + "/train-images-idx3-ubyte",
                data_dir_ + "/train-labels-idx1-ubyte");
}

MNISTDataset MNISTLoader::load_test() {
    return load(data_dir_ + "/t10k-images-idx3-ubyte",
                data_dir_ + "/t10k-labels-idx1-ubyte");
}

MNISTDataset MNISTLoader::load(const std::string& img_path,
                               const std::string& lbl_path) {
    std::ifstream img_f(img_path, std::ios::binary);
    std::ifstream lbl_f(lbl_path, std::ios::binary);

    if (!img_f) throw std::runtime_error("No se puede abrir: " + img_path);
    if (!lbl_f) throw std::runtime_error("No se puede abrir: " + lbl_path);

    uint32_t img_magic = read_u32_be(img_f);
    uint32_t lbl_magic = read_u32_be(lbl_f);
    if (img_magic != 0x803) throw std::runtime_error("Magic incorrecto en imagenes: " + img_path);
    if (lbl_magic != 0x801) throw std::runtime_error("Magic incorrecto en etiquetas: " + lbl_path);

    uint32_t n_imgs   = read_u32_be(img_f);
    uint32_t n_labels = read_u32_be(lbl_f);
    if (n_imgs != n_labels)
        throw std::runtime_error("Discrepancia entre cantidad de imagenes y etiquetas");

    uint32_t rows = read_u32_be(img_f);
    uint32_t cols = read_u32_be(img_f);
    int pixels = static_cast<int>(rows * cols);

    MNISTDataset ds;
    ds.num_samples = static_cast<int>(n_imgs);
    ds.images.resize(n_imgs, std::vector<float>(pixels));
    ds.labels.resize(n_imgs);

    std::vector<uint8_t> buf(pixels);
    for (uint32_t i = 0; i < n_imgs; ++i) {
        img_f.read(reinterpret_cast<char*>(buf.data()), pixels);
        for (int p = 0; p < pixels; ++p)
            ds.images[i][p] = static_cast<float>(buf[p]) / 255.f;
    }

    for (uint32_t i = 0; i < n_imgs; ++i) {
        uint8_t lbl;
        lbl_f.read(reinterpret_cast<char*>(&lbl), 1);
        ds.labels[i] = static_cast<int>(lbl);
    }

    return ds;
}

uint32_t MNISTLoader::read_u32_be(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16)
         | (uint32_t(bytes[2]) <<  8) |  uint32_t(bytes[3]);
}

} // namespace vit::utils
