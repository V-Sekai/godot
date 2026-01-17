#include "ufbxwt_extra_deflate.h"
#include <stdlib.h>

typedef size_t ufbxwte_deflate_compress_fn(ufbxwte_deflate_compressor *cc, void *dst, size_t dst_size, const void *src, size_t src_size);
typedef size_t ufbxwte_deflate_decompress_fn(ufbxwte_deflate_decompressor *dc, void *dst, size_t dst_size, const void *src, size_t src_size);
typedef size_t ufbxwte_deflate_free_fn(void *user);

#ifdef UFBXWT_HAS_ZLIB
#include <zlib.h>

static size_t ufbxwte_deflate_zlib_compress(ufbxwte_deflate_compressor *cc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
	uLong dst_len = (uLong)dst_size;
	if (compress2((Bytef*)dst, &dst_len, (const Bytef*)src, src_size, cc->level) != Z_OK) {
		return SIZE_MAX;
	}
	return dst_len;
}

static size_t ufbxwte_deflate_zlib_decompress(ufbxwte_deflate_decompressor *dc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
	uLong dst_len = (uLong)dst_size;
	if (uncompress((Bytef*)dst, &dst_len, (const Bytef*)src, src_size) != Z_OK) {
		return SIZE_MAX;
	}
	return dst_len;
}

static void ufbxwte_deflate_zlib_init_compressor(ufbxwte_deflate_compressor *cc, int32_t level)
{
	cc->compress_fn = &ufbxwte_deflate_zlib_compress;
}

static void ufbxwte_deflate_zlib_init_decompressor(ufbxwte_deflate_decompressor *dc)
{
	dc->decompress_fn = &ufbxwte_deflate_zlib_decompress;
}
#endif

#ifdef UFBXWT_HAS_LIBDEFLATE
#include <libdeflate.h>

static size_t ufbxwte_deflate_libdeflate_compress(ufbxwte_deflate_compressor *cc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
	size_t dst_len = libdeflate_zlib_compress((struct libdeflate_compressor*)cc->user, src, src_size, dst, dst_size);
	if (dst_len == 0) {
		return SIZE_MAX;
	}
	return dst_len;
}

static void ufbxwte_deflate_libdeflate_free_compressor(ufbxwte_deflate_compressor *cc)
{
	libdeflate_free_compressor((struct libdeflate_compressor*)cc->user);
}

static size_t ufbxwte_deflate_libdeflate_decompress(ufbxwte_deflate_decompressor *dc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
	size_t dst_len = 0;
	enum libdeflate_result result = libdeflate_zlib_decompress((struct libdeflate_decompressor*)dc->user, src, src_size, dst, dst_size, &dst_len);
	if (result != LIBDEFLATE_SUCCESS) {
		return SIZE_MAX;
	}
	return dst_len;
}

static void ufbxwte_deflate_libdeflate_free_decompressor(ufbxwte_deflate_decompressor *dc)
{
	libdeflate_free_decompressor((struct libdeflate_decompressor*)dc->user);
}

static void ufbxwte_deflate_libdeflate_init_compressor(ufbxwte_deflate_compressor *cc, int32_t level)
{
	cc->user = libdeflate_alloc_compressor(level);
	cc->compress_fn = &ufbxwte_deflate_libdeflate_compress;
	cc->free_fn = &ufbxwte_deflate_libdeflate_free_compressor;
}

static void ufbxwte_deflate_libdeflate_init_decompressor(ufbxwte_deflate_decompressor *dc)
{
	dc->user = libdeflate_alloc_decompressor();
	dc->decompress_fn = &ufbxwte_deflate_libdeflate_decompress;
	dc->free_fn = &ufbxwte_deflate_libdeflate_free_decompressor;
}
#endif

bool ufbxwte_deflate_init_compressor(ufbxwte_deflate_compressor *cc, ufbxwte_deflate_impl impl, int32_t level)
{
	switch (impl) {
#ifdef UFBXWT_HAS_ZLIB
	case UFBXWTE_DEFLATE_ZLIB:
		ufbxwte_deflate_zlib_init_compressor(cc, level);
		break;
#endif

#ifdef UFBXWT_HAS_LIBDEFLATE
	case UFBXWTE_DEFLATE_LIBDEFLATE:
		ufbxwte_deflate_libdeflate_init_compressor(cc, level);
		break;
#endif

	default:
		return false;
	}

	cc->impl = impl;
	cc->level = level;
	return true;
}

bool ufbxwte_deflate_init_decompressor(ufbxwte_deflate_decompressor *dc, ufbxwte_deflate_impl impl)
{
	switch (impl) {
#ifdef UFBXWT_HAS_ZLIB
	case UFBXWTE_DEFLATE_ZLIB:
		ufbxwte_deflate_zlib_init_decompressor(dc);
		break;
#endif

#ifdef UFBXWT_HAS_LIBDEFLATE
	case UFBXWTE_DEFLATE_LIBDEFLATE:
		ufbxwte_deflate_libdeflate_init_decompressor(dc);
		break;
#endif

	default:
		return false;
	}

	dc->impl = impl;
	return true;
}
