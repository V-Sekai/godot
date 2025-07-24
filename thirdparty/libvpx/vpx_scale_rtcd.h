/*
 * VPX Scale
 */

#pragma once

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct yv12_buffer_config;

#if CONFIG_SPATIAL_RESAMPLING
void vp8_horizontal_line_5_4_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_5_4_scale vp8_horizontal_line_5_4_scale_c

void vp8_vertical_band_5_4_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_5_4_scale vp8_vertical_band_5_4_scale_c

void vp8_horizontal_line_5_3_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_5_3_scale vp8_horizontal_line_5_3_scale_c

void vp8_vertical_band_5_3_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_5_3_scale vp8_vertical_band_5_3_scale_c

void vp8_horizontal_line_2_1_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_2_1_scale vp8_horizontal_line_2_1_scale_c

void vp8_vertical_band_2_1_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_2_1_scale vp8_vertical_band_2_1_scale_c

void vp8_vertical_band_2_1_scale_i_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_2_1_scale_i vp8_vertical_band_2_1_scale_i_c
#endif

void vp8_yv12_extend_frame_borders_c(struct yv12_buffer_config *ybf);
#define vp8_yv12_extend_frame_borders vp8_yv12_extend_frame_borders_c

void vp8_yv12_copy_frame_c(const struct yv12_buffer_config *src_ybc, struct yv12_buffer_config *dst_ybc);
#define vp8_yv12_copy_frame vp8_yv12_copy_frame_c

void vpx_yv12_copy_y_c(const struct yv12_buffer_config *src_ybc, struct yv12_buffer_config *dst_ybc);
#define vpx_yv12_copy_y vpx_yv12_copy_y_c

#if CONFIG_VP9
void vpx_yv12_copy_frame_c(const struct yv12_buffer_config *src_ybc, struct yv12_buffer_config *dst_ybc);
#define vpx_yv12_copy_frame vpx_yv12_copy_frame_c

void vpx_extend_frame_borders_c(struct yv12_buffer_config *ybf);
#define vpx_extend_frame_borders vpx_extend_frame_borders_c

void vpx_extend_frame_inner_borders_c(struct yv12_buffer_config *ybf);
#define vpx_extend_frame_inner_borders vpx_extend_frame_inner_borders_c
#endif

void vpx_scale_rtcd(void);

#ifdef __cplusplus
}  // extern "C"
#endif
