# GPU Backend Strategy

## Implementation

The LiteRT module implements the following backend priority strategy:

### Backend Priority Order (macOS)

1. **WebGPU** (Primary) - Tried first
   - Cross-platform backend for universal code path
   - Requires `libLiteRtWebGpuAccelerator.dylib` to be available
   - Uses Dawn library (Google's WebGPU implementation)

2. **Metal** (Fallback) - Tried second if WebGPU fails
   - Native macOS/iOS API (system framework)
   - No external dependencies required
   - Potentially better performance on macOS

3. **CPU** (Final Fallback) - Always available
   - XNNPACK-accelerated CPU kernels
   - Used if all GPU backends fail or are unavailable

## How It Works

The backend selection is handled by LiteRT's `auto_registration.cc`:

```cpp
// On macOS (non-Android, non-iOS):
static constexpr absl::string_view kGpuAcceleratorLibs[] = {
    "libLiteRtWebGpuAccelerator" SO_EXT,  // Tried FIRST
    "libLiteRtOpenClAccelerator" SO_EXT,
    "libLiteRtMetalAccelerator" SO_EXT,   // Tried THIRD (fallback)
};
```

The registration loop tries each backend in order and **stops at the first successful registration**:

```cpp
for (auto plugin_path : kGpuAcceleratorLibs) {
    auto registration = RegisterSharedObjectAccelerator(...);
    if (registration.HasValue()) {
        gpu_accelerator_registered = true;
        break;  // Stops here - WebGPU wins if available
    }
}
```

## Configuration

### Build Configuration

- **GPU Support**: Enabled (`LITERT_BUILD_CONFIG_DISABLE_GPU` is NOT defined)
- **WebGPU Support**: Enabled by default (`LITERT_HAS_WEBGPU_SUPPORT_DEFAULT 1`)
- **Metal Support**: Enabled on macOS (`LITERT_HAS_METAL_SUPPORT_DEFAULT 1` on `__APPLE__`)

### Source Files

- `runtime/gpu_environment.cc` - Always included (GPU environment support)
- `runtime/metal_info.cc` - Included on macOS/iOS (Metal support)

### Framework Linking

- Metal framework linked on macOS/iOS (system framework, no external dependency)

## Result

- **If WebGPU accelerator library is available**: WebGPU is used (cross-platform path)
- **If WebGPU fails/unavailable**: Metal is tried (native fallback)
- **If both fail**: CPU is used (XNNPACK-accelerated)

This provides:
- ✅ Cross-platform code path (WebGPU primary)
- ✅ Native performance when needed (Metal fallback)
- ✅ Reliability (multiple fallback options)

## Verification

To verify which backend is being used, check LiteRT logs:
- `"Dynamically loaded GPU accelerator(libLiteRtWebGpuAccelerator.dylib) registered."` = WebGPU
- `"Dynamically loaded GPU accelerator(libLiteRtMetalAccelerator.dylib) registered."` = Metal
- If neither appears, CPU backend is being used

