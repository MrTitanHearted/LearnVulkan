CXX := clang++
CXXFLAGS := -std=c++20 -O2 -Wall
TARGET := .\build\learnvulkan.exe
SOURCES := .\src\main.cpp
INC_DIR := -Iinclude -I$(VULKAN_SDK)\Include
INCLUDES := 
LIB_DIR := -Llib 
LDFLAGS := -lglfw3dll -lvulkan-1
DEBUG_DEFINES := -DDEBUG
RELEASE_DEFINES := -DNDEBUG

VERT_SHADER := .\shaders\shader.vert
FRAG_SHADER := .\shaders\shader.frag

build: $(TARGET)

$(TARGET): $(SOURCES) $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(DEBUG_DEFINES) $(SOURCES) -o $(TARGET) $(INC_DIR) $(LIB_DIR) $(LDFLAGS)

release:
	$(CXX) $(CXXFLAGS) $(RELEASE_DEFINES) $(SOURCES) -o $(TARGET) $(INC_DIR) $(LIB_DIR) $(LDFLAGS)

run:
	$(TARGET)

shader: $(VERT_SHADER) $(FRAG_SHADER)
	glslc $(VERT_SHADER) -o $(VERT_SHADER).spv
	glslc $(FRAG_SHADER) -o $(FRAG_SHADER).spv