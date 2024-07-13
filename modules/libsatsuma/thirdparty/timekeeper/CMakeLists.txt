cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18.0)

project (Timekeeper
    VERSION 0.0.1
    LANGUAGES C CXX
    )

if(NOT ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
    add_subdirectory(src)
else()

    if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
        message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
    endif()

    set(TIMEKEEPER_ENABLE_EXAMPLES   ON CACHE BOOL "Build Timekeeper examples")

    set(CMAKE_CXX_STANDARD 17)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON) # deprecated since CMake 3.16
    set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF) # since CMake 3.16

    add_subdirectory(src)

    if(${TIMEKEEPER_ENABLE_EXAMPLES})
        add_subdirectory(examples)
    endif()
endif()
