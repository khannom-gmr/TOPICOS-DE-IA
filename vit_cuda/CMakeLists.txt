cmake_minimum_required(VERSION 3.18)
project(vit_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# RTX 4070 Ti Super = Ada Lovelace = sm_89 (backward compatible with sm_50+)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

set(UTIL_SOURCES
    src/utils/logger.cpp
    src/utils/mnist_loader.cpp
    src/utils/metrics.cpp
)

add_library(vit_cuda_lib STATIC
    ${UTIL_SOURCES}
    src/cuda_tensor.cu
    src/cuda_layers.cu
    src/cuda_attention.cu
    src/cuda_transformer.cu
)
target_include_directories(vit_cuda_lib PUBLIC
    include
    ${CMAKE_SOURCE_DIR}/vit_cpp/include
)
target_link_libraries(vit_cuda_lib PUBLIC CUDA::cublas)

target_compile_options(vit_cuda_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -Wall>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    >
)

add_executable(vit_cuda apps/train_gpu.cpp)
target_link_libraries(vit_cuda PRIVATE vit_cuda_lib)
