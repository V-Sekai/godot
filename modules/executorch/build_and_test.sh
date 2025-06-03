#!/bin/bash

# Build and test script for ExecuTorchResource

set -e

echo "Building ExecuTorchResource tests..."

# Create build directory
mkdir -p build
cd build

# Compile the resource implementation
echo "Compiling ExecuTorchResource..."
g++ -std=c++17 -c ../src/executorch_resource.cpp -o executorch_resource.o

# Compile and run the simple test
echo "Compiling and running tests..."
g++ -std=c++17 -I../src ../tests/test_resource_simple.cpp executorch_resource.o -o test_resource

echo "Running ExecuTorchResource tests..."
./test_resource

echo "Tests completed successfully!"

# Clean up
cd ..
echo "Build artifacts in build/ directory"
