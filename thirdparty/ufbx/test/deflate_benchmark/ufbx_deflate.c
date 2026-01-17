#define UFBXW_UNIT_TEST 1
#define UFBXWI_FEATURE_ATOMICS 1
#define UFBXWI_FEATURE_THREAD_POOL 1
#define UFBXWI_FEATURE_ERROR 1
#define UFBXWI_FEATURE_DEFLATE 1
#include "../../ufbx_write.c"

#include "../ufbx/ufbx.h"

#include "ufbx_deflate.h"

static size_t ufbxwte_deflate_ufbx_compress(ufbxwte_deflate_compressor *cc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
	ufbxwi_deflate_encoder *ud = (ufbxwi_deflate_encoder*)cc->user;
	return ufbxwi_deflate(ud, dst, dst_size, src, src_size);
}

static void ufbxwte_deflate_ufbx_free_compressor(ufbxwte_deflate_compressor *cc)
{
	free(cc->user);
}

static size_t ufbxwte_deflate_ufbx_decompress(ufbxwte_deflate_decompressor *dc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
	ufbx_inflate_input input = { 0 };
	input.data = src;
	input.data_size = src_size;
	input.total_size = src_size;
	ptrdiff_t result = ufbx_inflate(dst, dst_size, &input, (ufbx_inflate_retain*)dc->user);
	if (result < 0) {
		return SIZE_MAX;
	}

	return (size_t)result;
}

static void ufbxwte_deflate_ufbx_free_decompressor(ufbxwte_deflate_decompressor *dc)
{
	free(dc->user);
}

bool ufbxwt_deflate_init_compressor(ufbxwte_deflate_compressor *cc, ufbxwte_deflate_impl impl, int32_t level)
{
	if (impl == UFBXWTE_DEFLATE_UFBX) {
		ufbxwi_deflate_encoder *ud = (ufbxwi_deflate_encoder*)malloc(sizeof(ufbxwi_deflate_encoder));
		ufbxwi_deflate_encoder_setup(ud, level);

		cc->user = ud;
		cc->compress_fn = &ufbxwte_deflate_ufbx_compress;
		cc->free_fn = &ufbxwte_deflate_ufbx_free_compressor;

		cc->level = level;
		cc->impl = impl;
		return true;
	} else {
		return ufbxwte_deflate_init_compressor(cc, impl, level);
	}
}

bool ufbxwt_deflate_init_decompressor(ufbxwte_deflate_decompressor *dc, ufbxwte_deflate_impl impl)
{
	if (impl == UFBXWTE_DEFLATE_UFBX) {
		ufbx_inflate_retain *retain = (ufbx_inflate_retain*)malloc(sizeof(ufbx_inflate_retain));
		if (!retain) return false;
		retain->initialized = false;

		dc->user = retain;
		dc->decompress_fn = &ufbxwte_deflate_ufbx_decompress;
		dc->free_fn = &ufbxwte_deflate_ufbx_free_decompressor;

		dc->impl = impl;
		return true;
	} else {
		return ufbxwte_deflate_init_decompressor(dc, impl);
	}
}
