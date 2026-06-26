cmake_minimum_required(VERSION 3.15)
project(MLP_CPP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(mlp_iris
    src/Matrix.cpp
    src/Layer.cpp
    src/Activations.cpp
    src/Loss.cpp
    src/MLP.cpp
    src/main.cpp
)
