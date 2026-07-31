#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vit::utils {

struct MNISTDataset {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    int num_samples = 0;

    float image_at(int sample, int pixel) const {
        return images[sample][pixel];
    }
};

class MNISTLoader {
public:
    explicit MNISTLoader(const std::string& data_dir);

    MNISTDataset load_train();
    MNISTDataset load_test();

    bool files_exist() const;

private:
    std::string data_dir_;

    MNISTDataset load(const std::string& img_path,
                      const std::string& lbl_path);

    static uint32_t read_u32_be(std::ifstream& f);
};

} // namespace vit::utils
