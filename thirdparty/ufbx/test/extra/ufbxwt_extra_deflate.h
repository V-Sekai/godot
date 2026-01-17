#ifndef UFBXW_EXTRA_DEFLATE_H
#define UFBXW_EXTRA_DEFLATE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
    UFBXWTE_DEFLATE_ZLIB,
    UFBXWTE_DEFLATE_LIBDEFLATE,
    UFBXWTE_DEFLATE_UFBX,

    UFBXWTE_DEFLATE_IMPL_COUNT,
} ufbxwte_deflate_impl;

typedef struct ufbxwte_deflate_compressor ufbxwte_deflate_compressor;
typedef struct ufbxwte_deflate_decompressor ufbxwte_deflate_decompressor;

typedef size_t ufbxwte_deflate_compress_fn(ufbxwte_deflate_compressor *cc, void *dst, size_t dst_size, const void *src, size_t src_size);
typedef void ufbxwte_deflate_free_compressor_fn(ufbxwte_deflate_compressor *cc);

typedef size_t ufbxwte_deflate_decompress_fn(ufbxwte_deflate_decompressor *dc, void *dst, size_t dst_size, const void *src, size_t src_size);
typedef void ufbxwte_deflate_free_decompressor_fn(ufbxwte_deflate_decompressor *dc);

struct ufbxwte_deflate_compressor {
    ufbxwte_deflate_impl impl;
    int32_t level;
    void *user;
    ufbxwte_deflate_compress_fn *compress_fn;
    ufbxwte_deflate_free_compressor_fn *free_fn;
};

struct ufbxwte_deflate_decompressor {
    ufbxwte_deflate_impl impl;
    void *user;
    ufbxwte_deflate_decompress_fn *decompress_fn;
    ufbxwte_deflate_free_decompressor_fn *free_fn;
};

bool ufbxwte_deflate_init_compressor(ufbxwte_deflate_compressor *cc, ufbxwte_deflate_impl impl, int32_t level);

static size_t ufbxwte_deflate_compress(ufbxwte_deflate_compressor *cc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
    return cc->compress_fn(cc, dst, dst_size, src, src_size);
}

static void ufbxwte_deflate_free_compressor(ufbxwte_deflate_compressor *cc)
{
    if (cc->free_fn) {
		cc->free_fn(cc);
    }
}

bool ufbxwte_deflate_init_decompressor(ufbxwte_deflate_decompressor *dc, ufbxwte_deflate_impl impl);

static size_t ufbxwte_deflate_decompress(ufbxwte_deflate_decompressor *dc, void *dst, size_t dst_size, const void *src, size_t src_size)
{
    return dc->decompress_fn(dc, dst, dst_size, src, src_size);
}

static void ufbxwte_deflate_free_decompressor(ufbxwte_deflate_decompressor *dc)
{
    if (dc->free_fn) {
		dc->free_fn(dc);
    }
}

#if defined(__cplusplus)
}
#endif

#endif
