CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
INCLUDES = -I include
SRCS = src/Matrix.cpp src/Layer.cpp src/Activations.cpp src/Loss.cpp src/MLP.cpp src/main.cpp
TARGET = mlp_iris

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
