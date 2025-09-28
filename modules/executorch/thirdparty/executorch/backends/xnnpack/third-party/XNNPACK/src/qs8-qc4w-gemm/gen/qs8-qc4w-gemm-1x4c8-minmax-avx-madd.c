// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-ssevnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <smmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__avx_madd(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const __m128i vsign_mask = _mm_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m128 voutput_max_less_zero_point = _mm_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m128i voutput_zero_point = _mm_set1_epi32(params->fp32_scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi16(params->fp32_scalar.output_min);
  const __m128i vmask = _mm_set1_epi8(0x0F);
  XNN_FORCE_REALIZATION(vmask);
  do {
    __m128i vacc0x01 = _mm_cvtepu32_epi64(_mm_loadu_si64(w));
    __m128i vacc0x23 = _mm_cvtepu32_epi64(_mm_loadu_si64(((const int32_t*) w + 2)));
    __m128i vacc1x0x01 = _mm_setzero_si128();
    __m128i vacc1x0x23 = _mm_setzero_si128();
    w = (const int32_t*) w + 4;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m128i va0x01234567 = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a0)), vsign_mask);
      const __m128i va0x89ABCDEF = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8)), vsign_mask);
      a0 += 16;

      const __m128i vbb01234567x0123 = _mm_load_si128(w);
      const __m128i vbb89ABCDEFx0123 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs01234567x23 = _mm_srli_epi32(vbb01234567x0123, 4);
      const __m128i vbs89ABCDEFx23 = _mm_srli_epi32(vbb89ABCDEFx0123, 4);
      const __m128i vb01234567x01 = _mm_and_si128(vbb01234567x0123, vmask);
      const __m128i vb89ABCDEFx01 = _mm_and_si128(vbb89ABCDEFx0123, vmask);
      const __m128i vb01234567x23 = _mm_and_si128(vbs01234567x23, vmask);
      const __m128i vb89ABCDEFx23 = _mm_and_si128(vbs89ABCDEFx23, vmask);

      vacc0x01 = _mm_dpbusd_epi32_madd(vacc0x01, va0x01234567, vb01234567x01);
      vacc0x23 = _mm_dpbusd_epi32_madd(vacc0x23, va0x01234567, vb89ABCDEFx01);
      vacc1x0x01 = _mm_dpbusd_epi32_madd(vacc1x0x01, va0x89ABCDEF, vb01234567x23);
      vacc1x0x23 = _mm_dpbusd_epi32_madd(vacc1x0x23, va0x89ABCDEF, vb89ABCDEFx23);

      w = (const int8_t*) w + 32;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m128i va0x01234567 = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a0)), vsign_mask);
      a0 += 8;

      const __m128i vbb01234567x0123 = _mm_load_si128(w);
      const __m128i vbb89ABCDEFx0123 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vb01234567x01 = _mm_and_si128(vbb01234567x0123, vmask);
      const __m128i vb89ABCDEFx01 = _mm_and_si128(vbb89ABCDEFx0123, vmask);

      vacc0x01 = _mm_dpbusd_epi32_madd(vacc0x01, va0x01234567, vb01234567x01);
      vacc0x23 = _mm_dpbusd_epi32_madd(vacc0x23, va0x01234567, vb89ABCDEFx01);

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }
    vacc0x01 = _mm_add_epi32(vacc0x01, vacc1x0x01);
    vacc0x23 = _mm_add_epi32(vacc0x23, vacc1x0x23);

    // Add adjacent pairs
    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps(w);
    w = (const float*) w + 4;
    vout0x0123 = _mm_mul_ps(vout0x0123, vscale0123);

    vout0x0123 = _mm_min_ps(vout0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vout0x0123);

    vacc0x0123 = _mm_add_epi32(vacc0x0123, voutput_zero_point);

    vacc0x0123 = _mm_packs_epi32(vacc0x0123, vacc0x0123);
    vacc0x0123 = _mm_max_epi16(vacc0x0123, voutput_min);
    __m128i voutb0x0123 = _mm_packs_epi16(vacc0x0123, vacc0x0123);

    if (nc >= 4) {
      _mm_storeu_si32(c0, voutb0x0123);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(voutb0x0123, 0));
        c0 += 2;
        voutb0x0123 = _mm_srli_epi32(voutb0x0123, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi16(voutb0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
