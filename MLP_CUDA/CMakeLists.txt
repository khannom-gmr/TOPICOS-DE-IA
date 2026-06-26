cmake_minimum_required(VERSION 3.18)
project(MLP_CUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070 Ti Super

add_executable(mlp_mnist
    src/kernels.cu
    src/CudaMatrix.cu
    src/MLP_CUDA.cu
    src/main.cu
    src/Matrix.cpp
    src/MLP_CPU.cpp
)

target_include_directories(mlp_mnist PRIVATE include)
set_target_properties(mlp_mnist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
