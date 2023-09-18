CXX := clang++
CXXFLAGS := -std=c++20 -Wall
TARGET := .\build\learnvulkan.exe
SOURCES := .\src\main.cpp
INC_DIR := -Iinclude
INCLUDES := 
LIB_DIR := -Llib
LDFLAGS := -lglfw3dll

build: $(TARGET)

$(TARGET): $(SOURCES) $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(INC_DIR) $(LIB_DIR) $(LDFLAGS)

run:
	$(TARGET)