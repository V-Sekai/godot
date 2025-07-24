/*
 * VP9
 */

#pragma once

#include "vpx/vpx_integer.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int16_t InterpKernel[8];

struct macroblockd;

/* Encoder forward decls */
struct macroblock;
struct vp9_variance_vtable;
struct search_site_config;
struct mv;
union int_mv;
struct yv12_buffer_config;

void vp9_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int stride, int tx_type);
#define vp9_iht4x4_16_add vp9_iht4x4_16_add_c

void vp9_iht8x8_64_add_c(const tran_low_t *input, uint8_t *dest, int stride, int tx_type);
#define vp9_iht8x8_64_add vp9_iht8x8_64_add_c

void vp9_iht16x16_256_add_c(const tran_low_t *input, uint8_t *dest, int stride, int tx_type);
#define vp9_iht16x16_256_add vp9_iht16x16_256_add_c

void vp9_idct4x4_1_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct4x4_1_add vp9_idct4x4_1_add_c

void vp9_idct4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct4x4_16_add vp9_idct4x4_16_add_c

void vp9_idct8x8_1_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct8x8_1_add vp9_idct8x8_1_add_c

void vp9_idct8x8_64_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct8x8_64_add vp9_idct8x8_64_add_c

void vp9_idct16x16_1_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct16x16_1_add vp9_idct16x16_1_add_c

void vp9_idct16x16_256_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct16x16_256_add vp9_idct16x16_256_add_c

void vp9_idct32x32_1_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct32x32_1_add vp9_idct32x32_1_add_c

void vp9_idct32x32_1024_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct32x32_1024_add vp9_idct32x32_1024_add_c

void vp9_idct32x32_135_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct32x32_135_add vp9_idct32x32_135_add_c

void vp9_idct32x32_34_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_idct32x32_34_add vp9_idct32x32_34_add_c

void vp9_iwht4x4_1_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_iwht4x4_1_add vp9_iwht4x4_1_add_c

void vp9_iwht4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int stride);
#define vp9_iwht4x4_16_add vp9_iwht4x4_16_add_c

void vp9_lpf_vertical_16_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_vertical_16 vp9_lpf_vertical_16_c

void vp9_lpf_vertical_16_dual_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_vertical_16_dual vp9_lpf_vertical_16_dual_c

void vp9_lpf_vertical_8_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_vertical_8 vp9_lpf_vertical_8_c

void vp9_lpf_vertical_8_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
#define vp9_lpf_vertical_8_dual vp9_lpf_vertical_8_dual_c

void vp9_lpf_vertical_4_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_vertical_4 vp9_lpf_vertical_4_c

void vp9_lpf_vertical_4_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
#define vp9_lpf_vertical_4_dual vp9_lpf_vertical_4_dual_c

void vp9_lpf_horizontal_16_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_horizontal_16 vp9_lpf_horizontal_16_c

void vp9_lpf_horizontal_16_dual_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_horizontal_16_dual vp9_lpf_horizontal_16_dual_c

void vp9_lpf_horizontal_8_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_horizontal_8 vp9_lpf_horizontal_8_c

void vp9_lpf_horizontal_8_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
#define vp9_lpf_horizontal_8_dual vp9_lpf_horizontal_8_dual_c

void vp9_lpf_horizontal_4_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
#define vp9_lpf_horizontal_4 vp9_lpf_horizontal_4_c

void vp9_lpf_horizontal_4_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
#define vp9_lpf_horizontal_4_dual vp9_lpf_horizontal_4_dual_c

void vp9_convolve_copy_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve_copy vp9_convolve_copy_c

void vp9_convolve_avg_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve_avg vp9_convolve_avg_c

void vp9_convolve8_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve8 vp9_convolve8_c

void vp9_convolve8_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve8_horiz vp9_convolve8_horiz_c

void vp9_convolve8_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve8_vert vp9_convolve8_vert_c

void vp9_convolve8_avg_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve8_avg vp9_convolve8_avg_c

void vp9_convolve8_avg_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve8_avg_horiz vp9_convolve8_avg_horiz_c

void vp9_convolve8_avg_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h);
#define vp9_convolve8_avg_vert vp9_convolve8_avg_vert_c

void vp9_d207_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d207_predictor_4x4 vp9_d207_predictor_4x4_c

void vp9_d45_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d45_predictor_4x4 vp9_d45_predictor_4x4_c

void vp9_d63_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d63_predictor_4x4 vp9_d63_predictor_4x4_c

void vp9_h_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_h_predictor_4x4 vp9_h_predictor_4x4_c

void vp9_d117_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d117_predictor_4x4 vp9_d117_predictor_4x4_c

void vp9_d135_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d135_predictor_4x4 vp9_d135_predictor_4x4_c

void vp9_d153_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d153_predictor_4x4 vp9_d153_predictor_4x4_c

void vp9_v_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_v_predictor_4x4 vp9_v_predictor_4x4_c

void vp9_tm_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_tm_predictor_4x4 vp9_tm_predictor_4x4_c

void vp9_dc_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_predictor_4x4 vp9_dc_predictor_4x4_c

void vp9_dc_top_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_top_predictor_4x4 vp9_dc_top_predictor_4x4_c

void vp9_dc_left_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_left_predictor_4x4 vp9_dc_left_predictor_4x4_c

void vp9_dc_128_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_128_predictor_4x4 vp9_dc_128_predictor_4x4_c

void vp9_d207_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d207_predictor_8x8 vp9_d207_predictor_8x8_c

void vp9_d45_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d45_predictor_8x8 vp9_d45_predictor_8x8_c

void vp9_d63_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d63_predictor_8x8 vp9_d63_predictor_8x8_c

void vp9_h_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_h_predictor_8x8 vp9_h_predictor_8x8_c

void vp9_d117_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d117_predictor_8x8 vp9_d117_predictor_8x8_c

void vp9_d135_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d135_predictor_8x8 vp9_d135_predictor_8x8_c

void vp9_d153_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d153_predictor_8x8 vp9_d153_predictor_8x8_c

void vp9_v_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_v_predictor_8x8 vp9_v_predictor_8x8_c

void vp9_tm_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_tm_predictor_8x8 vp9_tm_predictor_8x8_c

void vp9_dc_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_predictor_8x8 vp9_dc_predictor_8x8_c

void vp9_dc_top_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_top_predictor_8x8 vp9_dc_top_predictor_8x8_c

void vp9_dc_left_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_left_predictor_8x8 vp9_dc_left_predictor_8x8_c

void vp9_dc_128_predictor_8x8_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_128_predictor_8x8 vp9_dc_128_predictor_8x8_c

void vp9_d207_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d207_predictor_16x16 vp9_d207_predictor_16x16_c

void vp9_d45_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d45_predictor_16x16 vp9_d45_predictor_16x16_c

void vp9_d63_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d63_predictor_16x16 vp9_d63_predictor_16x16_c

void vp9_h_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_h_predictor_16x16 vp9_h_predictor_16x16_c

void vp9_d117_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d117_predictor_16x16 vp9_d117_predictor_16x16_c

void vp9_d135_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d135_predictor_16x16 vp9_d135_predictor_16x16_c

void vp9_d153_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d153_predictor_16x16 vp9_d153_predictor_16x16_c

void vp9_v_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_v_predictor_16x16 vp9_v_predictor_16x16_c

void vp9_tm_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_tm_predictor_16x16 vp9_tm_predictor_16x16_c

void vp9_dc_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_predictor_16x16 vp9_dc_predictor_16x16_c

void vp9_dc_top_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_top_predictor_16x16 vp9_dc_top_predictor_16x16_c

void vp9_dc_left_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_left_predictor_16x16 vp9_dc_left_predictor_16x16_c

void vp9_dc_128_predictor_16x16_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_128_predictor_16x16 vp9_dc_128_predictor_16x16_c

void vp9_d207_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d207_predictor_32x32 vp9_d207_predictor_32x32_c

void vp9_d45_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d45_predictor_32x32 vp9_d45_predictor_32x32_c

void vp9_d63_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d63_predictor_32x32 vp9_d63_predictor_32x32_c

void vp9_h_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_h_predictor_32x32 vp9_h_predictor_32x32_c

void vp9_d117_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d117_predictor_32x32 vp9_d117_predictor_32x32_c

void vp9_d135_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d135_predictor_32x32 vp9_d135_predictor_32x32_c

void vp9_d153_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_d153_predictor_32x32 vp9_d153_predictor_32x32_c

void vp9_v_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_v_predictor_32x32 vp9_v_predictor_32x32_c

void vp9_tm_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_tm_predictor_32x32 vp9_tm_predictor_32x32_c

void vp9_dc_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_predictor_32x32 vp9_dc_predictor_32x32_c

void vp9_dc_top_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_top_predictor_32x32 vp9_dc_top_predictor_32x32_c

void vp9_dc_left_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_left_predictor_32x32 vp9_dc_left_predictor_32x32_c

void vp9_dc_128_predictor_32x32_c(uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left);
#define vp9_dc_128_predictor_32x32 vp9_dc_128_predictor_32x32_c

unsigned int vp9_variance32x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance32x16 vp9_variance32x16_c

unsigned int vp9_variance16x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance16x32 vp9_variance16x32_c

unsigned int vp9_variance64x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance64x32 vp9_variance64x32_c

unsigned int vp9_variance32x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance32x64 vp9_variance32x64_c

unsigned int vp9_variance32x32_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance32x32 vp9_variance32x32_c

unsigned int vp9_variance64x64_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance64x64 vp9_variance64x64_c

unsigned int vp9_variance16x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance16x16 vp9_variance16x16_c

unsigned int vp9_variance16x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance16x8 vp9_variance16x8_c

unsigned int vp9_variance8x16_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance8x16 vp9_variance8x16_c

unsigned int vp9_variance8x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance8x8 vp9_variance8x8_c

unsigned int vp9_variance8x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance8x4 vp9_variance8x4_c

unsigned int vp9_variance4x8_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance4x8 vp9_variance4x8_c

unsigned int vp9_variance4x4_c(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
#define vp9_variance4x4 vp9_variance4x4_c

void vp9_rtcd(void);

#ifdef __cplusplus
}  // extern "C"
#endif
