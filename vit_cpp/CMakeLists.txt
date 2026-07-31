cmake_minimum_required(VERSION 3.18)
project(vit_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(UTIL_SOURCES
    src/utils/logger.cpp
    src/utils/mnist_loader.cpp
    src/utils/metrics.cpp
)

add_library(vit_cpu_lib STATIC
    ${UTIL_SOURCES}
    src/tensor.cpp
    src/layers.cpp
    src/attention.cpp
    src/transformer.cpp
    src/optimizer.cpp
)
target_include_directories(vit_cpu_lib PUBLIC include)
target_compile_options(vit_cpu_lib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
)

# Training executable
add_executable(vit_cpu apps/train_cpu.cpp)
target_link_libraries(vit_cpu PRIVATE vit_cpu_lib)

# Inference executable
add_executable(vit_infer apps/inference.cpp)
target_link_libraries(vit_infer PRIVATE vit_cpu_lib)

# Tests
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE vit_cpu_lib)
add_test(NAME TensorTests COMMAND test_tensor)

add_executable(test_attention tests/test_attention.cpp)
target_link_libraries(test_attention PRIVATE vit_cpu_lib)
add_test(NAME AttentionTests COMMAND test_attention)

add_executable(test_mnist tests/test_mnist_loader.cpp)
target_link_libraries(test_mnist PRIVATE vit_cpu_lib)
add_test(NAME MNISTLoader COMMAND test_mnist ${CMAKE_SOURCE_DIR}/../data)
