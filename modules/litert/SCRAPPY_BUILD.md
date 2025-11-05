# Scrappiest Build Guide - LiteRT Module

This guide outlines the minimal approach to get the LiteRT module working for matmul inference.

## Current Status

✅ **Completed:**
- pthreadpool: Built from source (8 C files)
- FlatBuffers: Built from source (4 C++ files)
- Abseil: Minimal build complete (~25 source files)
- XNNPACK: Skipped (optional acceleration)
- TFLite linking: Infrastructure ready

⚠️ **Remaining:**
- Build TensorFlow Lite separately and link pre-built library

## Quick Setup Steps

### 1. Build TensorFlow Lite (Required)

**Option A: Use the helper script (Easiest)**
```bash
# From project root
./modules/litert/build_tflite.sh
```

**Option B: Manual build**
```bash
cd thirdparty/tensorflow-lite
mkdir build && cd build

# Configure CMake
cmake ../tensorflow/lite -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF

# Build (this may take 10-30 minutes)
cmake --build . -j$(sysctl -n hw.ncpu)  # macOS
# Or: cmake --build . -j$(nproc)  # Linux

# Library is ready - SCons will find it automatically in the build directory
```

### 2. Verify Library Location

The SCsub build script will automatically find and link `libtensorflow-lite.a` from `thirdparty/tensorflow-lite/build/` when it exists.

### 3. Build Godot with LiteRT Module

```bash
# From project root
scons platform=linuxbsd target=template_debug
# Or: platform=macos, platform=windows, etc.
```

### 4. Create Test Model (Optional)

For the matmul test, you'll need a `matmul_model.tflite` file. You can:
- Create a simple 2x3 * 3x2 matrix multiplication model
- Download a minimal test model
- Place it at `res://test/matmul_model.tflite` (or update test path)

### 5. Run Tests

```bash
./bin/godot.* --headless --test --force-colors
```

## What's Skipped (For Speed)

- **XNNPACK**: Skipped entirely - TFLite runs without it (just slower)
- **Full Abseil**: Only minimal components built (~25 files vs 455 total)
- **TFLite in SCons**: Built separately with CMake instead

## Troubleshooting

### Link Errors
- Ensure `libtensorflow-lite.a` exists in `thirdparty/tensorflow-lite/build/`
- Check that library was built for the correct platform/architecture
- Verify all Abseil symbols are resolved (may need to add more files)

### Missing Symbols
- If you see undefined Abseil symbols, add the missing `.cc` files to `SCsub`
- Check Abseil BUILD.bazel files to see which components are needed

### TFLite Build Issues
- TensorFlow Lite can be complex to build
- Consider using a pre-built binary if available for your platform
- Check TensorFlow Lite documentation for platform-specific build requirements

## Next Steps (If Needed)

Once basic matmul works:
- Add XNNPACK for CPU acceleration (optional)
- Complete Abseil build if more components are needed
- Optimize build process

