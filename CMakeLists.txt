cmake_minimum_required(VERSION 3.20)
project(LearnVulkan)

set(CMAKE_CXX_COMPILER /home/linuxbrew/.linuxbrew/bin/clang++)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

add_subdirectory(glm)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(include)

set(TARGET LearnVulkan)

file(GLOB SOURCES "src/*.cpp")

add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET} glfw ${Vulkan_LIBRARIES} glm)