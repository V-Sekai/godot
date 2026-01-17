#include "unit_test.h"

#define UFBXW_UNIT_TEST 1
#define UFBXWI_FEATURE_DEFLATE 1
#include "../../ufbx_write.c"

#include "../ufbx/ufbx.h"

#define UFBXWT_UNIT_CATEGORY "deflate"

struct ufbxwt_deflate_symbol_info {
	uint32_t symbol;
	uint32_t extra_bits;
	uint32_t min_value;
	uint32_t max_value;
};

static const struct ufbxwt_deflate_symbol_info ufbxwt_deflate_distance_table[] = {
	{ 0, 0, 1, 1 }, 
	{ 1, 0, 2, 2 }, 
	{ 2, 0, 3, 3 }, 
	{ 3, 0, 4, 4 }, 
	{ 4, 1, 5, 6 }, 
	{ 5, 1, 7, 8 }, 
	{ 6, 2, 9, 12 }, 
	{ 7, 2, 13, 16 }, 
	{ 8, 3, 17, 24 }, 
	{ 9, 3, 25, 32 }, 
	{ 10, 4, 33, 48 }, 
	{ 11, 4, 49, 64 }, 
	{ 12, 5, 65, 96 }, 
	{ 13, 5, 97, 128 }, 
	{ 14, 6, 129, 192 }, 
	{ 15, 6, 193, 256 }, 
	{ 16, 7, 257, 384 }, 
	{ 17, 7, 385, 512 }, 
	{ 18, 8, 513, 768 }, 
	{ 19, 8, 769, 1024 }, 
	{ 20, 9, 1025, 1536 }, 
	{ 21, 9, 1537, 2048 }, 
	{ 22, 10, 2049, 3072 }, 
	{ 23, 10, 3073, 4096 }, 
	{ 24, 11, 4097, 6144 }, 
	{ 25, 11, 6145, 8192 }, 
	{ 26, 12, 8193, 12288 }, 
	{ 27, 12, 12289, 16384 }, 
	{ 28, 13, 16385, 24576 }, 
	{ 29, 13, 24577, 32768 }, 
};

static const struct ufbxwt_deflate_symbol_info ufbxwt_deflate_length_table[] = {
	{ 257, 0, 3, 3 }, 
	{ 258, 0, 4, 4 }, 
	{ 259, 0, 5, 5 }, 
	{ 260, 0, 6, 6 }, 
	{ 261, 0, 7, 7 }, 
	{ 262, 0, 8, 8 }, 
	{ 263, 0, 9, 9 }, 
	{ 264, 0, 10, 10 }, 
	{ 265, 1, 11, 12 }, 
	{ 266, 1, 13, 14 }, 
	{ 267, 1, 15, 16 }, 
	{ 268, 1, 17, 18 }, 
	{ 269, 2, 19, 22 }, 
	{ 270, 2, 23, 26 }, 
	{ 271, 2, 27, 30 }, 
	{ 272, 2, 31, 34 }, 
	{ 273, 3, 35, 42 }, 
	{ 274, 3, 43, 50 }, 
	{ 275, 3, 51, 58 }, 
	{ 276, 3, 59, 66 }, 
	{ 277, 4, 67, 82 }, 
	{ 278, 4, 83, 98 }, 
	{ 279, 4, 99, 114 }, 
	{ 280, 4, 115, 130 }, 
	{ 281, 5, 131, 162 }, 
	{ 282, 5, 163, 194 }, 
	{ 283, 5, 195, 226 }, 
	{ 284, 5, 227, 257 }, 
	{ 285, 0, 258, 258 }, 
};

UFBXWT_UNIT_TEST(deflate_saturate_chain_offset)
{
	ufbxwt_assert(ufbxwi_lz_saturate_chain_offset(100) == 100);
	ufbxwt_assert(ufbxwi_lz_saturate_chain_offset(1000) == 1000);
	ufbxwt_assert(ufbxwi_lz_saturate_chain_offset(10000) == 10000);
	ufbxwt_assert(ufbxwi_lz_saturate_chain_offset(100000) == 0xffff);
}

UFBXWT_UNIT_TEST(deflate_match_length)
{
	ufbxwt_assert(ufbxwi_lz_match_length_slow("Hello World Alpha", "Hello World Beta", 128) == 12);
	ufbxwt_assert(ufbxwi_lz_match_length_slow("Hello World Alpha", "Hello World Beta", 10) == 10);

	ufbxwt_assert(ufbxwi_lz_match_length_fast("Hello World Alpha........", "Hello World Beta........", 128) == 12);
	ufbxwt_assert(ufbxwi_lz_match_length_fast("Hello World Alpha........", "Hello World Beta........", 10) == 10);
}

UFBXWT_UNIT_TEST(deflate_length_to_symbol)
{
	for (ufbxwt_deflate_symbol_info info : ufbxwt_deflate_length_table) {
		for (uint32_t v = info.min_value; v <= info.max_value; v++) {
			uint32_t sym = ufbxwi_deflate_length_symbol(v);
			ufbxwt_assert(sym + 256 == info.symbol);
		}
	}
}

UFBXWT_UNIT_TEST(deflate_distance_to_symbol)
{
	for (ufbxwt_deflate_symbol_info info : ufbxwt_deflate_distance_table) {
		for (uint32_t v = info.min_value; v <= info.max_value; v++) {
			uint32_t sym = ufbxwi_deflate_dist_symbol(v);
			ufbxwt_assert(sym == info.symbol);
		}
	}
}

static void ufbxwt_test_deflate(const void *input, size_t input_length)
{
	const int32_t compression_level = 6;

	ufbxwi_deflate_encoder *ud = (ufbxwi_deflate_encoder*)malloc(sizeof(ufbxwi_deflate_encoder));
	ufbxwi_deflate_encoder_setup(ud, compression_level);

	size_t dst_size = input_length * 2 + 128;
	char *dst = (char*)malloc(dst_size);
	char *ref_dst = (char*)malloc(input_length);

	for (int32_t force_block_type = -1; force_block_type < 3; force_block_type++) {
		ud->force_block_type = (int8_t)force_block_type;

		size_t compressed_length = ufbxwi_deflate(ud, dst, dst_size, input, input_length);

		ufbx_inflate_retain ref_retain;
		ref_retain.initialized = false;

		ufbx_inflate_input ref_input = { 0 };
		ref_input.data = dst;
		ref_input.data_size = compressed_length;
		ref_input.total_size = compressed_length;

		ptrdiff_t result = ufbx_inflate(ref_dst, input_length, &ref_input, &ref_retain);
		ufbxwt_assert(result >= 0);
		ufbxwt_assert(result == input_length);
		ufbxwt_assert(!memcmp(input, ref_dst, input_length));
	}

	free(ref_dst);
	free(dst);
	free(ud);
}

static void ufbxwt_test_deflate_str(const char *str)
{
	ufbxwt_test_deflate(str, strlen(str));
}

UFBXWT_UNIT_TEST(deflate_simple)
{
	ufbxwt_test_deflate_str("Hello");
}

UFBXWT_UNIT_TEST(deflate_match)
{
	ufbxwt_test_deflate_str("Hello Hello");
}

UFBXWT_UNIT_TEST(deflate_rle)
{
	ufbxwt_test_deflate_str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
}

UFBXWT_UNIT_TEST(deflate_match_multi)
{
	ufbxwt_test_deflate_str("Hello World World Hello");
}

UFBXWT_UNIT_TEST(deflate_match_overlap)
{
	ufbxwt_test_deflate_str("HelloHello.. lloHe");
}

UFBXWT_UNIT_TEST(deflate_match_fast)
{
	char data[4096];
	for (size_t i = 0; i < 4096; i++)
	{
		data[i] = (char)i;
	}

	ufbxwt_test_deflate(data, sizeof(data));
}

