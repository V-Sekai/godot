#!/bin/bash
# Verification script to check if LiteRT module setup is complete

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== LiteRT Module Setup Verification ==="
echo ""

ERRORS=0
WARNINGS=0

# Check module files
echo "Checking module files..."
MODULE_FILES=(
    "modules/litert/config.py"
    "modules/litert/register_types.h"
    "modules/litert/register_types.cpp"
    "modules/litert/SCsub"
    "modules/litert/litrt_environment.h"
    "modules/litert/litrt_environment.cpp"
    "modules/litert/litrt_model.h"
    "modules/litert/litrt_model.cpp"
    "modules/litert/litrt_compiled_model.h"
    "modules/litert/litrt_compiled_model.cpp"
    "modules/litert/litrt_tensor_buffer.h"
    "modules/litert/litrt_tensor_buffer.cpp"
)

for file in "${MODULE_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ MISSING: $file"
        ((ERRORS++))
    fi
done

# Check dependencies
echo ""
echo "Checking dependencies..."

# pthreadpool
if [ -d "$PROJECT_ROOT/thirdparty/pthreadpool/src" ]; then
    PTHREADPOOL_FILES=$(find "$PROJECT_ROOT/thirdparty/pthreadpool/src" -name "*.c" ! -name "*test*" | wc -l | tr -d ' ')
    echo "  ✅ pthreadpool: $PTHREADPOOL_FILES source files"
else
    echo "  ❌ pthreadpool source directory missing"
    ((ERRORS++))
fi

# flatbuffers
if [ -d "$PROJECT_ROOT/thirdparty/flatbuffers/src" ]; then
    FLATBUFFERS_FILES=$(find "$PROJECT_ROOT/thirdparty/flatbuffers/src" -name "*.cpp" ! -name "*test*" ! -name "*benchmark*" | wc -l | tr -d ' ')
    echo "  ✅ flatbuffers: $FLATBUFFERS_FILES source files"
else
    echo "  ❌ flatbuffers source directory missing"
    ((ERRORS++))
fi

# abseil
if [ -d "$PROJECT_ROOT/thirdparty/abseil-cpp/absl" ]; then
    ABSEIL_FILES=$(find "$PROJECT_ROOT/thirdparty/abseil-cpp/absl" -name "*.cc" ! -name "*test*" ! -name "*benchmark*" | wc -l | tr -d ' ')
    echo "  ✅ abseil: $ABSEIL_FILES total .cc files available"
else
    echo "  ❌ abseil source directory missing"
    ((ERRORS++))
fi

# Check critical Abseil files
echo ""
echo "Checking critical Abseil files..."
ABSEIL_CRITICAL=(
    "absl/base/internal/raw_logging.cc"
    "absl/base/internal/spinlock.cc"
    "absl/status/status.cc"
    "absl/status/statusor.cc"
    "absl/strings/str_cat.cc"
    "absl/strings/str_format.cc"
    "absl/container/internal/raw_hash_set.cc"
)

for file in "${ABSEIL_CRITICAL[@]}"; do
    if [ -f "$PROJECT_ROOT/thirdparty/abseil-cpp/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ MISSING: $file"
        ((ERRORS++))
    fi
done

# Check TFLite
echo ""
echo "Checking TensorFlow Lite..."
if [ -f "$PROJECT_ROOT/thirdparty/tflite-libs/libtensorflow-lite.a" ]; then
    echo "  ✅ libtensorflow-lite.a found"
elif [ -d "$PROJECT_ROOT/thirdparty/tensorflow-lite" ]; then
    echo "  ⚠️  TFLite source found, but library not built yet"
    echo "     Run: ./modules/litert/build_tflite.sh"
    ((WARNINGS++))
else
    echo "  ⚠️  TFLite source directory missing"
    ((WARNINGS++))
fi

# Check helper scripts
echo ""
echo "Checking helper scripts..."
if [ -f "$PROJECT_ROOT/modules/litert/build_tflite.sh" ]; then
    echo "  ✅ build_tflite.sh"
else
    echo "  ❌ build_tflite.sh missing"
    ((ERRORS++))
fi

# Check documentation
echo ""
echo "Checking documentation..."
if [ -f "$PROJECT_ROOT/modules/litert/SCRAPPY_BUILD.md" ]; then
    echo "  ✅ SCRAPPY_BUILD.md"
else
    echo "  ⚠️  SCRAPPY_BUILD.md missing (optional)"
    ((WARNINGS++))
fi

# Summary
echo ""
echo "=== Summary ==="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All critical components found!"
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠️  $WARNINGS warning(s) - see above"
    fi
    echo ""
    echo "Next steps:"
    echo "1. Build TFLite: ./modules/litert/build_tflite.sh"
    echo "2. Build Godot: scons platform=macos target=template_debug"
    exit 0
else
    echo "❌ $ERRORS error(s) found - please fix before building"
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠️  $WARNINGS warning(s)"
    fi
    exit 1
fi

