#!/bin/bash
# Helper script to build TensorFlow Lite for the scrappiest approach
# This builds TFLite separately with CMake, then copies it to the expected location

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TFLITE_SOURCE="$PROJECT_ROOT/thirdparty/tensorflow-lite"
TFLITE_BUILD="$PROJECT_ROOT/thirdparty/tensorflow-lite/build"
TFLITE_LIBS="$PROJECT_ROOT/thirdparty/tflite-libs"

echo "Building TensorFlow Lite..."
echo "Source: $TFLITE_SOURCE"
echo "Build: $TFLITE_BUILD"
echo "Output: $TFLITE_LIBS"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed"
    echo "Install with: brew install cmake  (macOS) or apt-get install cmake (Linux)"
    exit 1
fi

# Check if source exists
if [ ! -d "$TFLITE_SOURCE" ]; then
    echo "Error: TensorFlow Lite source not found at $TFLITE_SOURCE"
    echo "Please ensure thirdparty/tensorflow-lite exists (git subrepo)"
    exit 1
fi

# Create build directory
mkdir -p "$TFLITE_BUILD"
cd "$TFLITE_BUILD"

# Configure CMake
echo "Configuring CMake..."
cmake "$TFLITE_SOURCE/tensorflow/lite" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_XNNPACK=OFF \
    || {
    echo "Error: CMake configuration failed"
    echo "Trying alternative: building from tensorflow root..."
    cd "$TFLITE_SOURCE"
    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTFLITE_ENABLE_GPU=OFF \
        -DTFLITE_ENABLE_XNNPACK=OFF \
        || {
        echo "Error: CMake configuration failed. You may need to:"
        echo "1. Install CMake (brew install cmake on macOS)"
        echo "2. Check TensorFlow Lite build documentation"
        exit 1
    }
}

# Build
echo "Building TensorFlow Lite (this may take 10-30 minutes)..."
cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) || {
    echo "Error: Build failed"
    exit 1
}

# Find the library
LIBRARY=""
if [ -f "libtensorflow-lite.a" ]; then
    LIBRARY="libtensorflow-lite.a"
elif [ -f "tensorflow/lite/libtensorflow-lite.a" ]; then
    LIBRARY="tensorflow/lite/libtensorflow-lite.a"
else
    echo "Warning: libtensorflow-lite.a not found in expected location"
    echo "Searching for it..."
    LIBRARY=$(find . -name "libtensorflow-lite.a" | head -1)
    if [ -z "$LIBRARY" ]; then
        echo "Error: Could not find libtensorflow-lite.a"
        exit 1
    fi
fi

# Copy to target location
mkdir -p "$TFLITE_LIBS"
cp "$LIBRARY" "$TFLITE_LIBS/libtensorflow-lite.a"

echo "✅ Success! libtensorflow-lite.a copied to $TFLITE_LIBS/"
echo ""
echo "Next steps:"
echo "1. Build Godot with: scons platform=macos target=template_debug"
echo "2. The module should now link successfully"

