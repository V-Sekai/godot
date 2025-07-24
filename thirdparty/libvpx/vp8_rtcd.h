/*
 * VP8
 */

#pragma once

#include "vpx/vpx_integer.h"
#include "vp8/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

struct blockd;
struct macroblockd;
struct loop_filter_info;

/* Encoder forward decls */
struct block;
struct macroblock;
struct variance_vtable;
union int_mv;
struct yv12_buffer_config;

void vp8_dequantize_b_c(struct blockd*, short *DQC);
#define vp8_dequantize_b vp8_dequantize_b_c

void vp8_dequant_idct_add_c(short *input, short *dq, unsigned char *dest, int stride);
#define vp8_dequant_idct_add vp8_dequant_idct_add_c

void vp8_dequant_idct_add_y_block_c(short *q, short *dq, unsigned char *dst, int stride, char *eobs);
#define vp8_dequant_idct_add_y_block vp8_dequant_idct_add_y_block_c

void vp8_dequant_idct_add_uv_block_c(short *q, short *dq, unsigned char *dst_u, unsigned char *dst_v, int stride, char *eobs);
#define vp8_dequant_idct_add_uv_block vp8_dequant_idct_add_uv_block_c

void vp8_loop_filter_mbv_c(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr, int y_stride, int uv_stride, struct loop_filter_info *lfi);
#define vp8_loop_filter_mbv vp8_loop_filter_mbv_c

void vp8_loop_filter_bv_c(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr, int y_stride, int uv_stride, struct loop_filter_info *lfi);
#define vp8_loop_filter_bv vp8_loop_filter_bv_c

void vp8_loop_filter_mbh_c(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr, int y_stride, int uv_stride, struct loop_filter_info *lfi);
#define vp8_loop_filter_mbh vp8_loop_filter_mbh_c

void vp8_loop_filter_bh_c(unsigned char *y_ptr, unsigned char *u_ptr, unsigned char *v_ptr, int y_stride, int uv_stride, struct loop_filter_info *lfi);
#define vp8_loop_filter_bh vp8_loop_filter_bh_c

void vp8_loop_filter_simple_mbv_c(unsigned char *y_ptr, int y_stride, const unsigned char *blimit);
#define vp8_loop_filter_simple_mbv vp8_loop_filter_simple_mbv_c

void vp8_loop_filter_simple_mbh_c(unsigned char *y_ptr, int y_stride, const unsigned char *blimit);
#define vp8_loop_filter_simple_mbh vp8_loop_filter_simple_mbh_c

void vp8_loop_filter_simple_bv_c(unsigned char *y_ptr, int y_stride, const unsigned char *blimit);
#define vp8_loop_filter_simple_bv vp8_loop_filter_simple_bv_c

void vp8_loop_filter_simple_bh_c(unsigned char *y_ptr, int y_stride, const unsigned char *blimit);
#define vp8_loop_filter_simple_bh vp8_loop_filter_simple_bh_c

void vp8_short_idct4x4llm_c(short *input, unsigned char *pred_ptr, int pred_stride, unsigned char *dst_ptr, int dst_stride);
#define vp8_short_idct4x4llm vp8_short_idct4x4llm_c

void vp8_short_inv_walsh4x4_c(short *input, short *mb_dqcoeff);
#define vp8_short_inv_walsh4x4 vp8_short_inv_walsh4x4_c

void vp8_short_inv_walsh4x4_1_c(short *input, short *mb_dqcoeff);
#define vp8_short_inv_walsh4x4_1 vp8_short_inv_walsh4x4_1_c

void vp8_dc_only_idct_add_c(short input_dc, unsigned char *pred_ptr, int pred_stride, unsigned char *dst_ptr, int dst_stride);
#define vp8_dc_only_idct_add vp8_dc_only_idct_add_c

void vp8_copy_mem16x16_c(unsigned char *src, int src_stride, unsigned char *dst, int dst_stride);
#define vp8_copy_mem16x16 vp8_copy_mem16x16_c

void vp8_copy_mem8x8_c(unsigned char *src, int src_stride, unsigned char *dst, int dst_stride);
#define vp8_copy_mem8x8 vp8_copy_mem8x8_c

void vp8_copy_mem8x4_c(unsigned char *src, int src_stride, unsigned char *dst, int dst_stride);
#define vp8_copy_mem8x4 vp8_copy_mem8x4_c

void vp8_build_intra_predictors_mby_s_c(struct macroblockd *x, unsigned char * yabove_row, unsigned char * yleft, int left_stride, unsigned char * ypred_ptr, int y_stride);
#define vp8_build_intra_predictors_mby_s vp8_build_intra_predictors_mby_s_c

void vp8_build_intra_predictors_mbuv_s_c(struct macroblockd *x, unsigned char * uabove_row, unsigned char * vabove_row,  unsigned char *uleft, unsigned char *vleft, int left_stride, unsigned char * upred_ptr, unsigned char * vpred_ptr, int pred_stride);
#define vp8_build_intra_predictors_mbuv_s vp8_build_intra_predictors_mbuv_s_c

void vp8_intra4x4_predict_c(unsigned char *Above, unsigned char *yleft, int left_stride, B_PREDICTION_MODE b_mode, unsigned char *dst, int dst_stride, unsigned char top_left);
#define vp8_intra4x4_predict vp8_intra4x4_predict_c

void vp8_sixtap_predict16x16_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_sixtap_predict16x16 vp8_sixtap_predict16x16_c

void vp8_sixtap_predict8x8_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_sixtap_predict8x8 vp8_sixtap_predict8x8_c

void vp8_sixtap_predict8x4_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_sixtap_predict8x4 vp8_sixtap_predict8x4_c

void vp8_sixtap_predict4x4_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_sixtap_predict4x4 vp8_sixtap_predict4x4_c

void vp8_bilinear_predict16x16_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_bilinear_predict16x16 vp8_bilinear_predict16x16_c

void vp8_bilinear_predict8x8_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_bilinear_predict8x8 vp8_bilinear_predict8x8_c

void vp8_bilinear_predict8x4_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_bilinear_predict8x4 vp8_bilinear_predict8x4_c

void vp8_bilinear_predict4x4_c(unsigned char *src_ptr, int src_pixels_per_line, int xoffset, int yoffset, unsigned char *dst_ptr, int dst_pitch);
#define vp8_bilinear_predict4x4 vp8_bilinear_predict4x4_c

unsigned int vp8_variance4x4_c(const unsigned char *src_ptr, int source_stride, const unsigned char *ref_ptr, int  ref_stride, unsigned int *sse);
#define vp8_variance4x4 vp8_variance4x4_c

unsigned int vp8_variance8x8_c(const unsigned char *src_ptr, int source_stride, const unsigned char *ref_ptr, int  ref_stride, unsigned int *sse);
#define vp8_variance8x8 vp8_variance8x8_c

unsigned int vp8_variance8x16_c(const unsigned char *src_ptr, int source_stride, const unsigned char *ref_ptr, int  ref_stride, unsigned int *sse);
#define vp8_variance8x16 vp8_variance8x16_c

unsigned int vp8_variance16x8_c(const unsigned char *src_ptr, int source_stride, const unsigned char *ref_ptr, int  ref_stride, unsigned int *sse);
#define vp8_variance16x8 vp8_variance16x8_c

unsigned int vp8_variance16x16_c(const unsigned char *src_ptr, int source_stride, const unsigned char *ref_ptr, int  ref_stride, unsigned int *sse);
#define vp8_variance16x16 vp8_variance16x16_c

unsigned int vp8_sub_pixel_variance4x4_c(const unsigned char  *src_ptr, int  source_stride, int  xoffset, int  yoffset, const unsigned char *ref_ptr, int Refstride, unsigned int *sse);
#define vp8_sub_pixel_variance4x4 vp8_sub_pixel_variance4x4_c

unsigned int vp8_sub_pixel_variance8x8_c(const unsigned char  *src_ptr, int  source_stride, int  xoffset, int  yoffset, const unsigned char *ref_ptr, int Refstride, unsigned int *sse);
#define vp8_sub_pixel_variance8x8 vp8_sub_pixel_variance8x8_c

unsigned int vp8_sub_pixel_variance8x16_c(const unsigned char  *src_ptr, int  source_stride, int  xoffset, int  yoffset, const unsigned char *ref_ptr, int Refstride, unsigned int *sse);
#define vp8_sub_pixel_variance8x16 vp8_sub_pixel_variance8x16_c

unsigned int vp8_sub_pixel_variance16x8_c(const unsigned char  *src_ptr, int  source_stride, int  xoffset, int  yoffset, const unsigned char *ref_ptr, int Refstride, unsigned int *sse);
#define vp8_sub_pixel_variance16x8 vp8_sub_pixel_variance16x8_c

unsigned int vp8_sub_pixel_variance16x16_c(const unsigned char  *src_ptr, int  source_stride, int  xoffset, int  yoffset, const unsigned char *ref_ptr, int Refstride, unsigned int *sse);
#define vp8_sub_pixel_variance16x16 vp8_sub_pixel_variance16x16_c

unsigned int vp8_get_mb_ss_c(const short *);
#define vp8_get_mb_ss vp8_get_mb_ss_c

unsigned int vp8_sub_pixel_mse16x16_c(const unsigned char  *src_ptr, int  source_stride, int  xoffset, int  yoffset, const unsigned char *ref_ptr, int Refstride, unsigned int *sse);
#define vp8_sub_pixel_mse16x16 vp8_sub_pixel_mse16x16_c

unsigned int vp8_mse16x16_c(const unsigned char *src_ptr, int source_stride, const unsigned char *ref_ptr, int  ref_stride, unsigned int *sse);
#define vp8_mse16x16 vp8_mse16x16_c

unsigned int vp8_get4x4sse_cs_c(const unsigned char *src_ptr, int source_stride, const unsigned char *ref_ptr, int  ref_stride);
#define vp8_get4x4sse_cs vp8_get4x4sse_cs_c

void vp8_short_fdct4x4_c(short *input, short *output, int pitch);
#define vp8_short_fdct4x4 vp8_short_fdct4x4_c

void vp8_short_fdct8x4_c(short *input, short *output, int pitch);
#define vp8_short_fdct8x4 vp8_short_fdct8x4_c

void vp8_short_walsh4x4_c(short *input, short *output, int pitch);
#define vp8_short_walsh4x4 vp8_short_walsh4x4_c

void vp8_regular_quantize_b_c(struct block *, struct blockd *);
#define vp8_regular_quantize_b vp8_regular_quantize_b_c

void vp8_fast_quantize_b_c(struct block *, struct blockd *);
#define vp8_fast_quantize_b vp8_fast_quantize_b_c

void vp8_regular_quantize_b_pair_c(struct block *b1, struct block *b2, struct blockd *d1, struct blockd *d2);
#define vp8_regular_quantize_b_pair vp8_regular_quantize_b_pair_c

void vp8_fast_quantize_b_pair_c(struct block *b1, struct block *b2, struct blockd *d1, struct blockd *d2);
#define vp8_fast_quantize_b_pair vp8_fast_quantize_b_pair_c

void vp8_quantize_mb_c(struct macroblock *);
#define vp8_quantize_mb vp8_quantize_mb_c

void vp8_quantize_mby_c(struct macroblock *);
#define vp8_quantize_mby vp8_quantize_mby_c

void vp8_quantize_mbuv_c(struct macroblock *);
#define vp8_quantize_mbuv vp8_quantize_mbuv_c

int vp8_block_error_c(short *coeff, short *dqcoeff);
#define vp8_block_error vp8_block_error_c

int vp8_mbblock_error_c(struct macroblock *mb, int dc);
#define vp8_mbblock_error vp8_mbblock_error_c

int vp8_mbuverror_c(struct macroblock *mb);
#define vp8_mbuverror vp8_mbuverror_c

void vp8_subtract_b_c(struct block *be, struct blockd *bd, int pitch);
#define vp8_subtract_b vp8_subtract_b_c

void vp8_subtract_mby_c(short *diff, unsigned char *src, int src_stride, unsigned char *pred, int pred_stride);
#define vp8_subtract_mby vp8_subtract_mby_c

void vp8_subtract_mbuv_c(short *diff, unsigned char *usrc, unsigned char *vsrc, int src_stride, unsigned char *upred, unsigned char *vpred, int pred_stride);
#define vp8_subtract_mbuv vp8_subtract_mbuv_c

int vp8_full_search_sad_c(struct macroblock *x, struct block *b, struct blockd *d, union int_mv *ref_mv, int sad_per_bit, int distance, struct variance_vtable *fn_ptr, int *mvcost[2], union int_mv *center_mv);
#define vp8_full_search_sad vp8_full_search_sad_c

int vp8_refining_search_sad_c(struct macroblock *x, struct block *b, struct blockd *d, union int_mv *ref_mv, int sad_per_bit, int distance, struct variance_vtable *fn_ptr, int *mvcost[2], union int_mv *center_mv);
#define vp8_refining_search_sad vp8_refining_search_sad_c

int vp8_diamond_search_sad_c(struct macroblock *x, struct block *b, struct blockd *d, union int_mv *ref_mv, union int_mv *best_mv, int search_param, int sad_per_bit, int *num00, struct variance_vtable *fn_ptr, int *mvcost[2], union int_mv *center_mv);
#define vp8_diamond_search_sad vp8_diamond_search_sad_c

void vp8_yv12_copy_partial_frame_c(struct yv12_buffer_config *src_ybc, struct yv12_buffer_config *dst_ybc);
#define vp8_yv12_copy_partial_frame vp8_yv12_copy_partial_frame_c

int vp8_denoiser_filter_c(unsigned char *mc_running_avg_y, int mc_avg_y_stride, unsigned char *running_avg_y, int avg_y_stride, unsigned char *sig, int sig_stride, unsigned int motion_magnitude, int increase_denoising);
#define vp8_denoiser_filter vp8_denoiser_filter_c

int vp8_denoiser_filter_uv_c(unsigned char *mc_running_avg, int mc_avg_stride, unsigned char *running_avg, int avg_stride, unsigned char *sig, int sig_stride, unsigned int motion_magnitude, int increase_denoising);
#define vp8_denoiser_filter_uv vp8_denoiser_filter_uv_c

void vp8_horizontal_line_4_5_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_4_5_scale vp8_horizontal_line_4_5_scale_c

void vp8_vertical_band_4_5_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_4_5_scale vp8_vertical_band_4_5_scale_c

void vp8_last_vertical_band_4_5_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_last_vertical_band_4_5_scale vp8_last_vertical_band_4_5_scale_c

void vp8_horizontal_line_2_3_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_2_3_scale vp8_horizontal_line_2_3_scale_c

void vp8_vertical_band_2_3_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_2_3_scale vp8_vertical_band_2_3_scale_c

void vp8_last_vertical_band_2_3_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_last_vertical_band_2_3_scale vp8_last_vertical_band_2_3_scale_c

void vp8_horizontal_line_3_5_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_3_5_scale vp8_horizontal_line_3_5_scale_c

void vp8_vertical_band_3_5_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_3_5_scale vp8_vertical_band_3_5_scale_c

void vp8_last_vertical_band_3_5_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_last_vertical_band_3_5_scale vp8_last_vertical_band_3_5_scale_c

void vp8_horizontal_line_3_4_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_3_4_scale vp8_horizontal_line_3_4_scale_c

void vp8_vertical_band_3_4_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_3_4_scale vp8_vertical_band_3_4_scale_c

void vp8_last_vertical_band_3_4_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_last_vertical_band_3_4_scale vp8_last_vertical_band_3_4_scale_c

void vp8_horizontal_line_1_2_scale_c(const unsigned char *source, unsigned int source_width, unsigned char *dest, unsigned int dest_width);
#define vp8_horizontal_line_1_2_scale vp8_horizontal_line_1_2_scale_c

void vp8_vertical_band_1_2_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_vertical_band_1_2_scale vp8_vertical_band_1_2_scale_c

void vp8_last_vertical_band_1_2_scale_c(unsigned char *source, unsigned int src_pitch, unsigned char *dest, unsigned int dest_pitch, unsigned int dest_width);
#define vp8_last_vertical_band_1_2_scale vp8_last_vertical_band_1_2_scale_c

void vp8_rtcd(void);

#ifdef __cplusplus
}  // extern "C"
#endif
