// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/microfnptr.h"

static struct xnn_ibilinear_config f16_ibilinear_config = {0};
static struct xnn_ibilinear_config f32_ibilinear_config = {0};
static struct xnn_ibilinear_config s8_ibilinear_config = {0};
static struct xnn_ibilinear_config u8_ibilinear_config = {0};

XNN_INIT_ONCE_GUARD(f16_ibilinear);
XNN_INIT_ONCE_GUARD(f32_ibilinear);
XNN_INIT_ONCE_GUARD(s8_ibilinear);
XNN_INIT_ONCE_GUARD(u8_ibilinear);

// Macros to log the microkernel names if and when they are registered.
#define XNN_INIT_IBILINEAR_UKERNEL(ukernel) \
  (xnn_ibilinear_ukernel_fn) ukernel;       \
  xnn_log_info("Using ibilinear microkernel '%s'.", #ukernel);

static void init_f16_ibilinear_config(void) {
  #if XNN_ENABLE_ARM_FP16_SCALAR && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if ((hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith)) {
      f16_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f16_ibilinear_ukernel__neonfp16arith_u8);
      f16_ibilinear_config.pixel_tile = 1;
    }
  #elif XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if ((hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith)) {
      f16_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f16_ibilinear_ukernel__neonfp16arith_u8);
      f16_ibilinear_config.pixel_tile = 1;
    }
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    #if XNN_ENABLE_FMA3
      if ((hardware_config->arch_flags & xnn_arch_x86_fma3)) {
        f16_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f16_ibilinear_ukernel__fma3_u8);
        f16_ibilinear_config.pixel_tile = 1;
      }
    #endif
  #endif
  f16_ibilinear_config.log2_data_element_size = XNN_LOG2_SIZEOF_HALF;
  f16_ibilinear_config.log2_weight_element_size = XNN_LOG2_SIZEOF_HALF;
  f16_ibilinear_config.indirection_init = (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_f16;
}

static void init_f32_ibilinear_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if ((hardware_config->arch_flags & xnn_arch_arm_neon)) {
      f32_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f32_ibilinear_ukernel__neon_u8);
      f32_ibilinear_config.pixel_tile = 1;
    } else {
      f32_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f32_ibilinear_ukernel__scalar_u2);
      f32_ibilinear_config.pixel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    f32_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f32_ibilinear_ukernel__neonfma_u8);
    f32_ibilinear_config.pixel_tile = 1;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    f32_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f32_ibilinear_ukernel__sse_u8);
    f32_ibilinear_config.pixel_tile = 1;
  #elif XNN_ARCH_WASMRELAXEDSIMD
    f32_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
    f32_ibilinear_config.pixel_tile = 1;
  #elif XNN_ARCH_WASMSIMD
    f32_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
    f32_ibilinear_config.pixel_tile = 1;
  #else
    f32_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_f32_ibilinear_ukernel__scalar_u2);
    f32_ibilinear_config.pixel_tile = 1;
  #endif
  f32_ibilinear_config.log2_data_element_size = XNN_LOG2_SIZEOF_FLOAT;
  f32_ibilinear_config.log2_weight_element_size = XNN_LOG2_SIZEOF_FLOAT;
  f32_ibilinear_config.indirection_init =
      (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_f32;
}

static void init_s8_ibilinear_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if ((hardware_config->arch_flags & xnn_arch_arm_neon)) {
      s8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_s8_ibilinear_ukernel__neon_u8);
      s8_ibilinear_config.pixel_tile = 1;
    } else {
      s8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_s8_ibilinear_ukernel__scalar_u1);
      s8_ibilinear_config.pixel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    s8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_s8_ibilinear_ukernel__neon_u16);
    s8_ibilinear_config.pixel_tile = 1;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if ((hardware_config->arch_flags & xnn_arch_x86_sse4_1)) {
      s8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_s8_ibilinear_ukernel__sse41_u16);
      s8_ibilinear_config.pixel_tile = 1;
    } else {
      s8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_s8_ibilinear_ukernel__sse2_u8);
      s8_ibilinear_config.pixel_tile = 1;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    s8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_u8);
    s8_ibilinear_config.pixel_tile = 1;
  #else
    s8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_s8_ibilinear_ukernel__scalar_u1);
    s8_ibilinear_config.pixel_tile = 1;
  #endif
  s8_ibilinear_config.log2_data_element_size = XNN_LOG2_SIZEOF_INT8_T;
  s8_ibilinear_config.log2_weight_element_size = XNN_LOG2_SIZEOF_INT16_T;
  s8_ibilinear_config.indirection_init =
      (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_q11;
}

static void init_u8_ibilinear_config(void) {
  #if XNN_ARCH_ARM
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if ((hardware_config->arch_flags & xnn_arch_arm_neon)) {
      u8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_u8_ibilinear_ukernel__neon_u8);
      u8_ibilinear_config.pixel_tile = 1;
    } else {
      u8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_u8_ibilinear_ukernel__scalar_u1);
      u8_ibilinear_config.pixel_tile = 1;
    }
  #elif XNN_ARCH_ARM64
    u8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_u8_ibilinear_ukernel__neon_u16);
    u8_ibilinear_config.pixel_tile = 1;
  #elif XNN_ARCH_X86 || XNN_ARCH_X86_64
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
    assert(hardware_config != NULL);
    (void) hardware_config;  // May be unused.
    if ((hardware_config->arch_flags & xnn_arch_x86_sse4_1)) {
      u8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_u8_ibilinear_ukernel__sse41_u16);
      u8_ibilinear_config.pixel_tile = 1;
    } else {
      u8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_u8_ibilinear_ukernel__sse2_u8);
      u8_ibilinear_config.pixel_tile = 1;
    }
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    u8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_u8_ibilinear_ukernel__wasmsimd_dot16x2_u8);
    u8_ibilinear_config.pixel_tile = 1;
  #else
    u8_ibilinear_config.ukernel = XNN_INIT_IBILINEAR_UKERNEL(xnn_u8_ibilinear_ukernel__scalar_u1);
    u8_ibilinear_config.pixel_tile = 1;
  #endif
  u8_ibilinear_config.log2_data_element_size = XNN_LOG2_SIZEOF_UINT8_T;
  u8_ibilinear_config.log2_weight_element_size = XNN_LOG2_SIZEOF_INT16_T;
  u8_ibilinear_config.indirection_init =
      (xnn_indirection_init_resize_bilinear2d_hwc_fn) xnn_indirection_init_resize_bilinear2d_hwc_q11;
}

const struct xnn_ibilinear_config* xnn_init_f16_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL || !xnn_is_f16_compatible_config(hardware_config)) {
    return NULL;
  }
  XNN_INIT_ONCE(f16_ibilinear);
  return &f16_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_f32_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(f32_ibilinear);
  return &f32_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_s8_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(s8_ibilinear);
  return &s8_ibilinear_config;
}

const struct xnn_ibilinear_config* xnn_init_u8_ibilinear_config() {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return NULL;
  }
  XNN_INIT_ONCE(u8_ibilinear);
  return &u8_ibilinear_config;
}
