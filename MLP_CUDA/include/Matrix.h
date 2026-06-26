#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>
#include <initializer_list>

class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(std::initializer_list<std::initializer_list<float>> values);

    float& operator()(int row, int col);
    float operator()(int row, int col) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(float scalar) const;

    // Producto matricial (dot product)
    Matrix multiply(const Matrix& other) const;
    Matrix transpose() const;

    // Aplica una función elemento a elemento
    Matrix apply(const std::function<float(float)>& fn) const;

    static Matrix zeros(int rows, int cols);
    static Matrix random(int rows, int cols, float min, float max);

    int rows() const;
    int cols() const;

private:
    int rows_;
    int cols_;
    std::vector<float> data_;
};

#endif
