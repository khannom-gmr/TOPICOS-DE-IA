#include "Matrix.h"
#include <random>
#include <stdexcept>

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(int rows, int cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> values) {
    rows_ = static_cast<int>(values.size());
    cols_ = rows_ > 0 ? static_cast<int>(values.begin()->size()) : 0;
    data_.reserve(rows_ * cols_);
    for (const auto& row : values) {
        for (float v : row) {
            data_.push_back(v);
        }
    }
}

float& Matrix::operator()(int row, int col) {
    return data_[row * cols_ + col];
}

float Matrix::operator()(int row, int col) const {
    return data_[row * cols_ + col];
}

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Dimensiones incompatibles en multiply");
    }
    Matrix result(rows_, other.cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int k = 0; k < cols_; ++k) {
            float a = data_[i * cols_ + k];
            for (int j = 0; j < other.cols_; ++j) {
                result.data_[i * other.cols_ + j] += a * other.data_[k * other.cols_ + j];
            }
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result.data_[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return result;
}

Matrix Matrix::apply(const std::function<float(float)>& fn) const {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i) {
        result.data_[i] = fn(data_[i]);
    }
    return result;
}

Matrix Matrix::zeros(int rows, int cols) {
    return Matrix(rows, cols);
}

// Inicializar pesos con distribución uniforme y seed fija
Matrix Matrix::random(int rows, int cols, float min, float max) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min, max);
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        result.data_[i] = dist(gen);
    }
    return result;
}

int Matrix::rows() const { return rows_; }
int Matrix::cols() const { return cols_; }
