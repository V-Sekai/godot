cmake_minimum_required(VERSION 3.15)
project(opcode_to_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include sandbox CMake configuration for RISC-V toolchain
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../sandbox/program/cpp/cmake/CMakeLists.txt")

# Get API directory from sandbox module
set(API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../sandbox/program/cpp/docker/api")

# Include directories for sandbox API
include_directories(${API_DIR})

# Source files
set(SOURCES
    src/main.cpp
    src/stablehlo_parser.cpp
    src/cpp_generator.cpp
    src/riscv_compiler.cpp
)

# Create executable
add_executable(opcode_to_cpp ${SOURCES})

# Link against sandbox API sources (we'll compile them directly)
target_sources(opcode_to_cpp PRIVATE
    ${API_DIR}/api.cpp
    ${API_DIR}/array.cpp
    ${API_DIR}/basis.cpp
    ${API_DIR}/dictionary.cpp
    ${API_DIR}/native.cpp
    ${API_DIR}/node.cpp
    ${API_DIR}/node2d.cpp
    ${API_DIR}/node3d.cpp
    ${API_DIR}/object.cpp
    ${API_DIR}/packed_array.cpp
    ${API_DIR}/quaternion.cpp
    ${API_DIR}/string.cpp
    ${API_DIR}/timer.cpp
    ${API_DIR}/transform2d.cpp
    ${API_DIR}/transform3d.cpp
    ${API_DIR}/variant.cpp
    ${API_DIR}/vector.cpp
)

# Compile definitions
target_compile_definitions(opcode_to_cpp PRIVATE
    VERSION=1.0
)

# Apply RISC-V compiler flags from sandbox CMake
target_compile_options(opcode_to_cpp PRIVATE
    ${CXXFLAGS}
    ${RISCV_ARCH}
    ${RISCV_ABI}
    -Wall
    -Wextra
)

# Link flags for RISC-V
target_link_options(opcode_to_cpp PRIVATE
    -static
    -nostdlib
    ${RISCV_ARCH}
    ${RISCV_ABI}
)

# Fetch stablehlo-translate as external dependency
include(FetchContent)
FetchContent_Declare(
    stablehlo
    GIT_REPOSITORY https://github.com/openxla/stablehlo.git
    GIT_TAG v0.11.0
    SOURCE_SUBDIR stablehlo
)

# Only fetch if not already available
if(NOT TARGET stablehlo-translate)
    FetchContent_MakeAvailable(stablehlo)
endif()

