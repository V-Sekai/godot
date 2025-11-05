# Complete LiteRT Module Integration

## Current Status

### Completed
- Module structure: `modules/litert/` with all wrapper classes
- Wrapper classes: `LiteRtEnvironment`, `LiteRtModel`, `LiteRtCompiledModel`, `LiteRtTensorBuffer`
- Build configuration: `SCsub` with dependencies (pthreadpool, FlatBuffers, Abseil minimal)
- Test infrastructure: C++ doctest with programmatic model generation
- TFLite CMake configuration: Successfully configured (just completed)

### Remaining Tasks
1. Build TensorFlow Lite library (10-30 minutes)
2. Copy library to expected location
3. Build Godot with litert module
4. Verify compilation succeeds
5. Run matmul test and verify it works

## Implementation Steps

### Step 1: Build TensorFlow Lite
**Location**: `thirdparty/tensorflow-lite/build/`

**Note**: Fixed CMake bug in `CMakeLists.txt` line 882 - added conditional check for `xnnpack-delegate` target.

CMake configuration succeeded. Build the core library:
```bash
cd thirdparty/tensorflow-lite/build
cmake --build . -j$(sysctl -n hw.ncpu) --target tensorflow-lite
```

This will compile the TensorFlow Lite library (may take 10-30 minutes). XNNPACK is disabled since we don't need it.

### Step 2: Verify Library Location
**Location**: `thirdparty/tensorflow-lite/build/libtensorflow-lite.a` (or `thirdparty/tensorflow-lite/build/tensorflow/lite/libtensorflow-lite.a`)

The `SCsub` file automatically finds the library in the build directory. No copy step needed - SCons will link it directly from where CMake builds it.

### Step 3: Build Godot with LiteRT Module
**Location**: Project root

```bash
scons platform=macos target=template_debug
```

This should:
- Compile LiteRT wrapper classes
- Compile dependency sources (pthreadpool, FlatBuffers, Abseil)
- Link against `libtensorflow-lite.a`
- Produce `bin/godot.macos.template_debug*`

### Step 4: Verify Build Success
Check for:
- No linker errors
- Module classes registered in Godot
- Binary created successfully

### Step 5: Run Matmul Test
**Test file**: `modules/litert/tests/test_matmul.cpp`

The test:
- Generates a matmul model programmatically using TFLite ModelBuilder
- Creates environment, model, compiled model
- Runs inference with 2x3 input matrix
- Verifies output is non-zero

```bash
./bin/godot.* --headless --test --force-colors
```

## Files to Modify

None - all code is complete. Only build steps remain.

## Potential Issues

1. **CMake bug**: Fixed `CMakeLists.txt` line 882 - xnnpack-delegate target check needed
2. **Link errors**: If TFLite symbols are missing, verify library was built correctly
3. **Missing Abseil symbols**: May need to add more Abseil source files to `SCsub`
4. **Model generation fails**: Test falls back to file-based approach with helpful error message
5. **Platform-specific issues**: CMake flags may need adjustment for different platforms
6. **XNNPACK disabled**: Build without XNNPACK for simplicity (optional acceleration)

## Success Criteria

- TFLite library built in `thirdparty/tensorflow-lite/build/` (SCons finds it automatically)
- Godot builds successfully with litert module
- Matmul test passes (or provides clear error message if model generation fails)
