cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18.0)

project (satsuma
    VERSION 0.0.2
    LANGUAGES C CXX
    )

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(STANDALONE_BUILD TRUE)
else()
    set(STANDALONE_BUILD FALSE)
endif()

set(SATSUMA_ENABLE_TESTS ${STANDALONE_BUILD} CACHE BOOL "Build satsuma tests")
set(SATSUMA_ENABLE_TOOLS ${STANDALONE_BUILD} CACHE BOOL "Build satsuma examples/tools")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON) # deprecated since CMake 3.16
set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF) # since CMake 3.16

set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")

add_subdirectory(src)
add_subdirectory(external/lemon)
add_subdirectory(external/timekeeper)

if(${SATSUMA_ENABLE_TESTS})
    enable_testing()
    add_subdirectory(tests)
endif()
if(${SATSUMA_ENABLE_TOOLS})
    add_subdirectory(tools)
endif()
