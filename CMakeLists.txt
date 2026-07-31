cmake_minimum_required(VERSION 3.18)
project(mnist_vit_root LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "==================================================")
message(STATUS "Vision Transformer (ViT) MNIST - Proyectos C++ y CUDA")
message(STATUS "==================================================")

# Subproyecto C++ CPU (Siempre habilitado)
add_subdirectory(vit_cpp)

# Subproyecto CUDA GPU (Opcional - requiere NVCC / CUDA Toolkit)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Compilador CUDA detectado: ${CMAKE_CUDA_COMPILER}")
    add_subdirectory(vit_cuda)
else()
    message(STATUS "Compilador CUDA no detectado - Se compila unicamente la version C++ CPU.")
endif()
