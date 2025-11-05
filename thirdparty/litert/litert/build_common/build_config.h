// Generated build_config.h for GPU-enabled build (WebGPU, Metal, CPU)
// Defines LiteRT feature toggles based on the selected build options.

#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_

// Enable GPU support (includes WebGPU and Metal)
// #define LITERT_BUILD_CONFIG_DISABLE_GPU 1  // Commented out to enable GPU
#define LITERT_BUILD_CONFIG_DISABLE_NPU 1

// GPU is enabled, so LITERT_DISABLE_GPU is not defined
// This enables WebGPU and Metal backends

#if LITERT_BUILD_CONFIG_DISABLE_NPU
#define LITERT_DISABLE_NPU
#endif  // LITERT_BUILD_CONFIG_DISABLE_NPU

#endif  // LITERT_BUILD_COMMON_BUILD_CONFIG_H_
