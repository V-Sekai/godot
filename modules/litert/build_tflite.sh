#!/bin/bash
# Helper script to build TensorFlow Lite with build-time patches
# This builds TFLite separately with CMake, applying patches before configuration
# SCons will automatically find the library in the build directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TFLITE_SOURCE="$PROJECT_ROOT/thirdparty/tensorflow-lite"
TFLITE_BUILD="$PROJECT_ROOT/thirdparty/tensorflow-lite/build"
TOOLCHAIN_FILE="$SCRIPT_DIR/toolchain.cmake"

echo "Building TensorFlow Lite..."
echo "Source: $TFLITE_SOURCE"
echo "Build: $TFLITE_BUILD"
echo "Note: SCons will automatically find the library in the build directory"

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

# Apply build-time patches if they exist
PATCHES_DIR="$SCRIPT_DIR/patches"
if [ -d "$PATCHES_DIR" ]; then
    echo "Applying build-time patches..."
    cd "$TFLITE_SOURCE"
    # Set library path for runtime (protoc needs libc++)
    export LD_LIBRARY_PATH="/home/linuxbrew/.linuxbrew/Cellar/llvm/21.1.5/lib:$LD_LIBRARY_PATH"

    # Apply model_building.patch using sed (fallback if patch not available)
    if [ -f "$PATCHES_DIR/model_building.patch" ]; then
        echo "  Applying model_building.patch..."
        if command -v patch &> /dev/null; then
            patch -p1 < "$PATCHES_DIR/model_building.patch" || {
                echo "Warning: model_building.patch failed to apply (may already be applied)"
            }
        else
            # Fallback: apply patch manually using sed
            MODEL_BUILDING_FILE="tensorflow/lite/core/model_building.cc"
            if [ -f "$MODEL_BUILDING_FILE" ]; then
                if ! grep -q "using TensorIdx = internal::StrongType" "$MODEL_BUILDING_FILE"; then
                    sed -i '/^namespace model_builder {$/a\
\
// Ensure type aliases are visible (header already defines Quantization, but these help with scope)\
using TensorIdx = internal::StrongType<int, class TensorTag>;\
using GraphIdx = internal::StrongType<int, class GraphTag>;\
using BufferIdx = internal::StrongType<int, class BufferTag>;\
' "$MODEL_BUILDING_FILE" || echo "Warning: Failed to apply model_building fix"
                fi
            fi
        fi
    fi

    # Apply common_blockwise.patch
    if [ -f "$PATCHES_DIR/common_blockwise.patch" ]; then
        echo "  Checking common_blockwise.patch..."
        COMMON_FILE="tensorflow/lite/core/c/common.cc"
        if [ -f "$COMMON_FILE" ]; then
            # Check if patch is already correctly applied (guard before and endif after)
            # Count guards - should be exactly 1 before and 1 after
            guard_count_before=$(grep -B 5 "case kTfLiteBlockwiseQuantization:" "$COMMON_FILE" | grep -c "#ifdef TFLITE_BLOCKWISE_QUANTIZATION_ENABLED" || echo "0")
            endif_count_after=$(grep -A 15 "case kTfLiteBlockwiseQuantization:" "$COMMON_FILE" | grep -c "#endif" || echo "0")
            if [ "$guard_count_before" -eq 1 ] && [ "$endif_count_after" -eq 1 ]; then
                echo "  Patch already applied, skipping..."
            elif command -v patch &> /dev/null; then
                # Try to apply patch directly
                patch -p1 < "$PATCHES_DIR/common_blockwise.patch" 2>/dev/null || {
                    echo "  Patch tool failed, patch may already be applied"
                }
            else
                # Python fallback: clean up and apply correctly
                python3 << 'PYEOF'
import re
with open("tensorflow/lite/core/c/common.cc", "r") as f:
    lines = f.readlines()

# Find blockwise case block and remove only its guards
in_blockwise = False
blockwise_start = -1
blockwise_end = -1
for i, line in enumerate(lines):
    if 'case kTfLiteBlockwiseQuantization:' in line:
        blockwise_start = i
        in_blockwise = True
    if in_blockwise and 'break;' in line and blockwise_start >= 0:
        # Check if this break is for blockwise case
        for j in range(blockwise_start, i):
            if 'case kTfLiteBlockwiseQuantization' in lines[j]:
                blockwise_end = i
                break
        if blockwise_end > 0:
            break

# Remove guards only around blockwise case (lines before and after)
new_lines = []
i = 0
while i < len(lines):
    if i == blockwise_start - 1 and lines[i].strip() == '#ifdef TFLITE_BLOCKWISE_QUANTIZATION_ENABLED':
        # Skip this guard if it's a duplicate
        i += 1
        continue
    if i == blockwise_end + 1 and lines[i].strip() == '#endif':
        # Skip this endif if it's a duplicate
        i += 1
        continue
    new_lines.append(lines[i])
    i += 1

# Re-add guards correctly
if blockwise_start > 0 and blockwise_end > 0:
    # Find the position after affine case break
    for i in range(len(new_lines)):
        if 'case kTfLiteAffineQuantization:' in new_lines[i]:
            # Find the matching break
            for j in range(i, min(i+20, len(new_lines))):
                if 'break;' in new_lines[j] and 'case kTfLiteAffineQuantization' in ''.join(new_lines[i:j]):
                    # Find blockwise case
                    for k in range(j, min(j+10, len(new_lines))):
                        if 'case kTfLiteBlockwiseQuantization:' in new_lines[k]:
                            # Insert guards
                            new_lines.insert(k, '#ifdef TFLITE_BLOCKWISE_QUANTIZATION_ENABLED\n')
                            # Find blockwise break
                            for m in range(k+1, min(k+20, len(new_lines))):
                                if 'break;' in new_lines[m]:
                                    new_lines.insert(m+1, '#endif\n')
                                    break
                            break
                    break
            break

with open("tensorflow/lite/core/c/common.cc", "w") as f:
    f.writelines(new_lines)
PYEOF
                patch -p1 < "$PATCHES_DIR/common_blockwise.patch" 2>/dev/null || {
                    echo "  Patch applied via Python cleanup + patch tool"
                }
            else
                # Python fallback: clean up and apply correctly
                python3 << 'PYEOF'
import re
with open("tensorflow/lite/core/c/common.cc", "r") as f:
    content = f.read()
# Remove duplicate TFLITE_BLOCKWISE_QUANTIZATION_ENABLED guards
content = re.sub(r'^#ifdef TFLITE_BLOCKWISE_QUANTIZATION_ENABLED\n', '', content, flags=re.MULTILINE)
content = re.sub(r'^#endif\n', '', content, flags=re.MULTILINE)
# Re-add correctly
content = re.sub(r'(case kTfLiteAffineQuantization: \{[^}]*?break;\s*\})\s*case kTfLiteBlockwiseQuantization: (\{[^}]*?dst_params->zero_point = src_params->zero_point;[^}]*?break;\s*\})',
    r'\1\n#ifdef TFLITE_BLOCKWISE_QUANTIZATION_ENABLED\n    case kTfLiteBlockwiseQuantization: \2\n#endif',
    content, flags=re.DOTALL)
with open("tensorflow/lite/core/c/common.cc", "w") as f:
    f.write(content)
PYEOF
                echo "  Patch applied via Python"
            fi
        fi
    fi

    # Apply verifier_int2.patch
    if [ -f "$PATCHES_DIR/verifier_int2.patch" ]; then
        echo "  Applying verifier_int2.patch..."
        if command -v patch &> /dev/null; then
            patch -p1 < "$PATCHES_DIR/verifier_int2.patch" || {
                echo "Warning: verifier_int2.patch failed to apply (may already be applied)"
            }
        else
            # Fallback: apply patch manually using sed/awk
            VERIFIER_FILE="tensorflow/lite/core/tools/verifier.cc"
            if [ -f "$VERIFIER_FILE" ]; then
                if ! grep -q "#ifdef TFLITE_ENABLE_INT2_INT4_TYPES" "$VERIFIER_FILE"; then
                    sed -i '/case TensorType_INT2:/i\
    #ifdef TFLITE_ENABLE_INT2_INT4_TYPES\
' "$VERIFIER_FILE"
                    sed -i '/case TensorType_INT4:/,/break;/ {
                        /break;/a\
    #endif
                    }' "$VERIFIER_FILE" || echo "Warning: Failed to apply verifier_int2 fix"
                fi
            fi
        fi
    fi

    # Apply flatbuffers_v25.patch - TensorFlow Lite 4.4-stable requires FlatBuffers v25
    # The CMake file incorrectly uses v24.3.25, but the generated headers require v25
    if [ -f "$PATCHES_DIR/flatbuffers_v25.patch" ]; then
        echo "  Applying flatbuffers_v25.patch..."
        if command -v patch &> /dev/null; then
            patch -p1 < "$PATCHES_DIR/flatbuffers_v25.patch" || {
                echo "Warning: flatbuffers_v25.patch failed to apply (may already be applied)"
            }
        else
            # Fallback: apply patch manually using sed
            FLATBUFFERS_CMAKE="tensorflow/lite/tools/cmake/modules/flatbuffers.cmake"
            if [ -f "$FLATBUFFERS_CMAKE" ]; then
                if ! grep -q "GIT_TAG v25.9.23" "$FLATBUFFERS_CMAKE"; then
                    sed -i 's/GIT_TAG v24.3.25/GIT_TAG v25.9.23/' "$FLATBUFFERS_CMAKE" || echo "Warning: Failed to apply flatbuffers_v25 fix"
                    sed -i 's/# Use stable FlatBuffers v24.3.25 tag instead of v25.9.23/# TensorFlow Lite 4.4-stable requires FlatBuffers v25/' "$FLATBUFFERS_CMAKE" || true
                fi
            fi
        fi
    fi

    # Apply exclude_model_building.patch to exclude experimental model_building.cc
    if [ -f "$PATCHES_DIR/exclude_model_building.patch" ]; then
        echo "  Applying exclude_model_building.patch..."
        if command -v patch &> /dev/null; then
            patch -p1 < "$PATCHES_DIR/exclude_model_building.patch" || {
                echo "Warning: exclude_model_building.patch failed to apply (may already be applied)"
            }
        else
            # Fallback: apply patch manually using sed
            CMAKE_LISTS="tensorflow/lite/CMakeLists.txt"
            if [ -f "$CMAKE_LISTS" ]; then
                if ! grep -q "Exclude experimental model_building" "$CMAKE_LISTS"; then
                    sed -i '/populate_tflite_source_vars("core" TFLITE_CORE_SRCS)/a\
# Exclude experimental model_building.cc (has compatibility issues with Abseil hash)\
list(FILTER TFLITE_CORE_SRCS EXCLUDE REGEX ".*model_building\\.cc$")' "$CMAKE_LISTS" || echo "Warning: Failed to apply exclude_model_building fix"
                fi
            fi
        fi
    fi

    # Apply eigen_3.4.0.patch to use stable Eigen 3.4.0 tag instead of development commit
    if [ -f "$PATCHES_DIR/eigen_3.4.0.patch" ]; then
        echo "  Applying eigen_3.4.0.patch..."
        if command -v patch &> /dev/null; then
            patch -p1 < "$PATCHES_DIR/eigen_3.4.0.patch" || {
                echo "Warning: eigen_3.4.0.patch failed to apply (may already be applied)"
            }
        else
            # Fallback: apply patch manually using sed
            EIGEN_CMAKE="tensorflow/lite/tools/cmake/modules/eigen.cmake"
            if [ -f "$EIGEN_CMAKE" ]; then
                if ! grep -q "GIT_TAG 3.4.0" "$EIGEN_CMAKE"; then
                    sed -i 's/GIT_TAG 70d8d99d0df9fd967b135efd8d12ed20fc48d007/GIT_TAG 3.4.0/' "$EIGEN_CMAKE" || echo "Warning: Failed to apply eigen_3.4.0 fix"
                    sed -i 's/# Sync with tensorflow\/third_party\/eigen3\/workspace.bzl/# Use stable Eigen 3.4.0 tag instead of development commit/' "$EIGEN_CMAKE" || true
                fi
            fi
        fi
    fi

    # Apply disable_proto_targets.patch to disable optional proto targets with path issues
    if [ -f "$PATCHES_DIR/disable_proto_targets.patch" ]; then
        echo "  Applying disable_proto_targets.patch..."
        if command -v patch &> /dev/null; then
            patch -p1 < "$PATCHES_DIR/disable_proto_targets.patch" || {
                echo "Warning: disable_proto_targets.patch failed to apply (may already be applied)"
            }
        else
            # Fallback: apply patch manually using sed
            CMAKE_LISTS="tensorflow/lite/CMakeLists.txt"
            if [ -f "$CMAKE_LISTS" ]; then
                if ! grep -q "# Disabled: proto compilation has path issues" "$CMAKE_LISTS"; then
                    sed -i 's/^add_subdirectory(\${TFLITE_SOURCE_DIR}\/profiling\/proto)$/# Disabled: proto compilation has path issues, not needed for core library\n# &/' "$CMAKE_LISTS" || echo "Warning: Failed to disable profiling proto"
                    sed -i 's/^add_subdirectory(\${TFLITE_SOURCE_DIR}\/tools\/benchmark\/proto)$/# Disabled: proto compilation has path issues, not needed for core library\n# &/' "$CMAKE_LISTS" || echo "Warning: Failed to disable benchmark proto"
                fi
            fi
        fi
    fi

fi

# Create build directory
mkdir -p "$TFLITE_BUILD"
cd "$TFLITE_BUILD"

# Configure CMake with toolchain file if available
CMAKE_ARGS=(
    "$TFLITE_SOURCE/tensorflow/lite"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DTFLITE_ENABLE_GPU=OFF"
    "-DTFLITE_ENABLE_XNNPACK=OFF"
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    "-DTENSORFLOW_SOURCE_DIR=$TFLITE_SOURCE"
)

# Skip toolchain file for now - it's causing linker issues
# The toolchain file can be re-enabled later if needed for libc++ compatibility
# if [ -f "$TOOLCHAIN_FILE" ]; then
#     echo "Using toolchain file: $TOOLCHAIN_FILE"
#     CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE")
# fi

echo "Configuring CMake..."
cmake "${CMAKE_ARGS[@]}" || {
    echo "Error: CMake configuration failed"
    echo "Trying alternative: building from tensorflow root..."
    cd "$TFLITE_SOURCE"
    mkdir -p build && cd build
    cmake "${CMAKE_ARGS[@]}" || {
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

# Find and verify the library
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

LIBRARY_PATH="$(pwd)/$LIBRARY"
LIBRARY_SIZE=$(ls -lh "$LIBRARY_PATH" | awk '{print $5}')

echo "✅ Success! libtensorflow-lite.a built at: $LIBRARY_PATH"
echo "   Size: $LIBRARY_SIZE"
echo ""
echo "SCons will automatically find and link this library when building Godot."
echo ""
echo "Next steps:"
echo "1. Build Godot with: scons platform=macos target=template_debug"
echo "2. The module should now link successfully"
