#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>
#include <initializer_list>

class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    // Producto matricial
    Matrix multiply(const Matrix& other) const;
    Matrix transpose() const;

    // Aplica una función elemento a elemento
    Matrix apply(const std::function<double(double)>& fn) const;

    static Matrix zeros(int rows, int cols);
    static Matrix random(int rows, int cols, double min, double max);

    int rows() const;
    int cols() const;

private:
    int rows_;
    int cols_;
    std::vector<double> data_;
};

#endif
