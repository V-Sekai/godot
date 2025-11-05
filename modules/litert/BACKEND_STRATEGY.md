# GPU Backend Strategy

## Implementation

The LiteRT module implements the following backend priority strategy:

### Backend Priority Order

1. **WebGPU** (Primary) - Tried first
   - Cross-platform backend for universal code path
   - Requires `libLiteRtWebGpuAccelerator.dylib` (or `.so`/`.dll`) to be available
   - Uses Dawn library (Google's WebGPU implementation)

2. **CPU** (Final Fallback) - Always available
   - XNNPACK-accelerated CPU kernels
   - Used if WebGPU backend fails or is unavailable

## How It Works

The backend selection is handled by LiteRT's `auto_registration.cc`:

```cpp
// On macOS (non-Android, non-iOS):
static constexpr absl::string_view kGpuAcceleratorLibs[] = {
    "libLiteRtWebGpuAccelerator" SO_EXT,  // Tried FIRST
    "libLiteRtOpenClAccelerator" SO_EXT,
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
- **Metal Support**: Disabled (removed from build)

### Source Files

- `runtime/gpu_environment.cc` - Always included (GPU environment support)

### Framework Linking

- No Metal framework linking (WebGPU-only strategy)

## Result

- **If WebGPU accelerator library is available**: WebGPU is used (cross-platform path)
- **If WebGPU fails/unavailable**: CPU is used (XNNPACK-accelerated)

This provides:
- ✅ Cross-platform code path (WebGPU primary)
- ✅ Simpler build (no platform-specific Metal dependencies)
- ✅ Reliability (CPU fallback always available)

## Verification

To verify which backend is being used, check LiteRT logs:
- `"Dynamically loaded GPU accelerator(libLiteRtWebGpuAccelerator.dylib) registered."` = WebGPU
- If this doesn't appear, CPU backend is being used

