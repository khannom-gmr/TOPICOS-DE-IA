NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_89 -std=c++17
# sm_89 = RTX 4070 Ti Super (Ada Lovelace)
INCLUDES = -I include
SRCS_CU = src/kernels.cu src/CudaMatrix.cu src/MLP_CUDA.cu src/main.cu
SRCS_CPP = src/Matrix.cpp src/MLP_CPU.cpp
TARGET = mlp_mnist

all: $(TARGET)

$(TARGET): $(SRCS_CU) $(SRCS_CPP)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRCS_CU) $(SRCS_CPP) -o $(TARGET)

clean:
	rm -f $(TARGET)
