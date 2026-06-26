#include "Matrix.h"
#include "Layer.h"
#include "MLP.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>

// Cargar iris.csv parseando manualmente
void loadIris(const std::string& path,
              std::vector<std::vector<double>>& features,
              std::vector<int>& labels) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir " << path << std::endl;
        return;
    }

    std::map<std::string, int> classMap = {
        {"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}
    };

    std::string line;
    std::getline(file, line); // cabecera
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, cell, ',');
            row.push_back(std::stod(cell));
        }
        std::getline(ss, cell, ',');
        features.push_back(row);
        labels.push_back(classMap[cell]);
    }
}

// Normalizar features con min-max scaling
void normalize(std::vector<std::vector<double>>& features) {
    int cols = features[0].size();
    for (int j = 0; j < cols; ++j) {
        double minV = features[0][j], maxV = features[0][j];
        for (const auto& row : features) {
            minV = std::min(minV, row[j]);
            maxV = std::max(maxV, row[j]);
        }
        double range = maxV - minV;
        for (auto& row : features) {
            row[j] = range > 0 ? (row[j] - minV) / range : 0.0;
        }
    }
}

int main() {
    std::vector<std::vector<double>> features;
    std::vector<int> labels;
    loadIris("data/iris.csv", features, labels);

    if (features.empty()) {
        std::cerr << "Dataset vacio." << std::endl;
        return 1;
    }

    normalize(features);

    // Shuffle con seed fija para reproducibilidad
    std::vector<int> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(42);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Dividir datos 80% train / 20% test
    int total = features.size();
    int trainSize = static_cast<int>(total * 0.8);

    Matrix Xtrain(trainSize, 4), Ytrain(trainSize, 3);
    Matrix Xtest(total - trainSize, 4), Ytest(total - trainSize, 3);

    for (int i = 0; i < total; ++i) {
        int idx = indices[i];
        bool isTrain = i < trainSize;
        Matrix& X = isTrain ? Xtrain : Xtest;
        Matrix& Y = isTrain ? Ytrain : Ytest;
        int r = isTrain ? i : i - trainSize;

        for (int j = 0; j < 4; ++j) X(r, j) = features[idx][j];
        // One-hot encode de la etiqueta
        for (int j = 0; j < 3; ++j) Y(r, j) = (labels[idx] == j) ? 1.0 : 0.0;
    }

    // Crear MLP con arquitectura 4-8-6-3
    std::vector<Layer> layers;
    layers.emplace_back(4, 8, "relu");
    layers.emplace_back(8, 6, "relu");
    layers.emplace_back(6, 3, "softmax");
    MLP model(std::move(layers));

    model.train(Xtrain, Ytrain, 500, 0.01, 16);

    double testAcc = model.computeAccuracy(Xtest, Ytest);
    std::cout << "\n[Test] Accuracy: " << std::fixed << std::setprecision(2)
              << testAcc << "%" << std::endl;

    return 0;
}
