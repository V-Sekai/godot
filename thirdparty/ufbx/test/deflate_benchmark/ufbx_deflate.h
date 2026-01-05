#ifndef UFBX_DEFLATE_H_INCLUDED
#define UFBX_DEFLATE_H_INCLUDED

#include <stdbool.h>
#include <stdint.h>

#include "../extra/ufbxwt_extra_deflate.h"

#if defined(__cplusplus)
extern "C" {
#endif

bool ufbxwt_deflate_init_compressor(ufbxwte_deflate_compressor *cc, ufbxwte_deflate_impl impl, int32_t level);
bool ufbxwt_deflate_init_decompressor(ufbxwte_deflate_decompressor *dc, ufbxwte_deflate_impl impl);

#if defined(__cplusplus)
}
#endif

#endif
