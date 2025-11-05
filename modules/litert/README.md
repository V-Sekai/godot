# LiteRT Module for Godot

This module provides GDScript bindings for [LiteRT](https://ai.google.dev/edge/litert) (formerly TensorFlow Lite), Google's high-performance runtime for on-device AI.

## Quick Start

### 1. Build TensorFlow Lite

Use the helper script:
```bash
./modules/litert/build_tflite.sh
```

Or see [SCRAPPY_BUILD.md](SCRAPPY_BUILD.md) for manual instructions.

### 2. Build Godot

```bash
scons platform=macos target=template_debug
# Or: platform=linuxbsd, platform=windows, etc.
```

### 3. Use in GDScript

```gdscript
var env = LiteRtEnvironment.new()
env.create()

var model = LiteRtModel.new()
model.load_from_file("res://path/to/model.tflite")

var compiled_model = LiteRtCompiledModel.new()
compiled_model.create(env, model)

# Create input tensor buffer
var input_buffer = LiteRtTensorBuffer.new()
input_buffer.create_from_array(input_data, input_shape)

# Run inference
var outputs = [output_buffer]
compiled_model.run(0, [input_buffer], outputs)
```

## API

### LiteRtEnvironment
- `create() -> Error`: Initialize the LiteRT environment

### LiteRtModel
- `load_from_file(path: String) -> Error`: Load a TFLite model from file
- `is_valid() -> bool`: Check if model is loaded

### LiteRtCompiledModel
- `create(environment: LiteRtEnvironment, model: LiteRtModel) -> Error`: Compile model
- `run(execution_id: int, inputs: Array, outputs: Array) -> Error`: Run inference

### LiteRtTensorBuffer
- `create_from_array(data: PackedFloat32Array, shape: PackedInt32Array) -> Error`: Create buffer from array
- `get_data() -> PackedFloat32Array`: Get buffer data

## GPU Backends

The LiteRT module supports multiple GPU backends:

### CPU (Always Available)
- **XNNPACK**: CPU acceleration via XNNPACK kernels
- Always available as fallback

### Metal (macOS/iOS)
- **Framework**: Metal (system framework, no external dependency)
- **Availability**: macOS and iOS platforms
- **Status**: Enabled when GPU support is enabled
- **Source**: `runtime/metal_info.cc` compiled as Objective-C++

### WebGPU (Cross-Platform)
- **Library**: Requires `libLiteRtWebGpuAccelerator.dylib` (or `.so`/`.dll`)
- **Dependency**: Dawn library (Google's WebGPU implementation)
- **Availability**: All platforms when accelerator library is available
- **Status**: Enabled when GPU support is enabled
- **Note**: Accelerator library must be built separately or provided

### Backend Priority Order

On macOS, LiteRT tries accelerators in this order:
1. **WebGPU** (`libLiteRtWebGpuAccelerator.dylib`) - **Primary choice**
   - Cross-platform backend (works on macOS, Linux, Windows, Android, Web)
   - Universal backend that can target all platforms
   - Requires Dawn library and accelerator library to be built
2. **Metal** (native) - **Fallback**
   - Native macOS/iOS API (system framework, no external dependency)
   - Potentially better performance on macOS due to native optimization
   - Reliable fallback if WebGPU is unavailable
3. **CPU** (XNNPACK) - **Always available**
   - Final fallback if GPU backends fail
   - XNNPACK-accelerated CPU kernels

**Rationale**: WebGPU is prioritized for cross-platform compatibility, while Metal serves as a reliable native fallback. This provides the best of both worlds: universal code path with native performance when needed.

## Dependencies

- **pthreadpool**: Built from source (8 files)
- **FlatBuffers**: Built from source (4 files)
- **Abseil**: Minimal build (~25 files)
- **TensorFlow Lite**: Pre-built library (build separately)
- **XNNPACK**: Skipped (optional acceleration)
- **Metal Framework**: System framework (macOS/iOS only)
- **WebGPU Accelerator**: Built separately (requires Dawn library)

## Build Status

✅ Module structure complete
✅ Wrapper classes implemented
✅ Dependencies configured
⚠️ TensorFlow Lite needs to be built separately

See [SCRAPPY_BUILD.md](SCRAPPY_BUILD.md) for detailed build instructions.
