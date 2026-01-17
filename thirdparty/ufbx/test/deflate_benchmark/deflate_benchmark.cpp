#include "../util/cputime.h"
#include "../util/im_arg.h"
#include "../util/json_input.h"
#include "ufbx_deflate.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>

static void ufbxwt_assert_fail(const char *file, uint32_t line, const char *expr) {
	fprintf(stderr, "assert fail: %s (%s:%u)\n", expr, file, line);
	exit(1);
}

#define ufbxwt_assert(cond) do { \
		if (!(cond)) ufbxwt_assert_fail(__FILE__, __LINE__, #cond); \
	} while (0)

bool g_verbose;

struct test_array
{
	const char *array_path;
	char type;
	uint32_t uncompressed_size;
	uint32_t compressed_size;
	uint32_t offset;
};

struct test_file
{
	const char *source_path;
	const char *data_path;
	size_t data_size;
	std::vector<test_array> arrays;
};

std::vector<char> g_src_buffer;
std::vector<char> g_dst_buffer;
std::vector<char> g_compress_buffer;
std::vector<char> g_check_buffer;

const char *deflate_impl_name[] = {
	"zlib",
	"libd",
	"ufbx",
};

struct benchmark_decompress_result {
	uint64_t min_delta[UFBXWTE_DEFLATE_IMPL_COUNT];
	uint64_t max_delta[UFBXWTE_DEFLATE_IMPL_COUNT];
	size_t uncompressed_bytes;
	size_t compressed_bytes;
};

benchmark_decompress_result benchmark_decompress(const test_file &file)
{
	benchmark_decompress_result result = { };
	ufbxwte_deflate_decompressor dc[UFBXWTE_DEFLATE_IMPL_COUNT] = { };

	for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
		ufbxwt_deflate_init_decompressor(&dc[impl], (ufbxwte_deflate_impl)impl);
		result.min_delta[impl] = UINT64_MAX;
		result.max_delta[impl] = 0;
	}

	{
		FILE *f = fopen(file.data_path, "rb");
		size_t num_read = fread(g_src_buffer.data(), 1, file.data_size, f);
		ufbxwt_assert(num_read == file.data_size);
		fclose(f);
	}

	for (const test_array &arr : file.arrays) {
		result.uncompressed_bytes += arr.uncompressed_size;
		result.compressed_bytes += arr.compressed_size;
	}

	for (size_t round = 0; round < 3; round++) {
		for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
			const uint64_t pre = cputime_cpu_tick();
			for (const test_array &arr : file.arrays) {
				void *dst = g_dst_buffer.data();
				size_t dst_size = arr.uncompressed_size;

				const void *src = g_src_buffer.data() + arr.offset;
				size_t src_size = arr.compressed_size;

				ufbxwte_deflate_decompress(&dc[impl], dst, dst_size, src, src_size);
			}
			const uint64_t post = cputime_cpu_tick();
			const uint64_t delta = post - pre;

			if (delta < result.min_delta[impl]) {
				result.min_delta[impl] = delta;
			}
			if (delta < result.max_delta[impl]) {
				result.max_delta[impl] = delta;
			}
		}
	}

	for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
		ufbxwte_deflate_free_decompressor(&dc[impl]);
	}

	return result;
}

struct benchmark_compress_result {
	uint64_t min_delta[UFBXWTE_DEFLATE_IMPL_COUNT];
	uint64_t max_delta[UFBXWTE_DEFLATE_IMPL_COUNT];
	size_t compressed_bytes[UFBXWTE_DEFLATE_IMPL_COUNT];
	size_t uncompressed_bytes;
};

benchmark_compress_result benchmark_compress(const test_file &file, int32_t level)
{
	benchmark_compress_result result = { };
	ufbxwte_deflate_compressor cc[UFBXWTE_DEFLATE_IMPL_COUNT] = { };

	for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
		ufbxwt_deflate_init_compressor(&cc[impl], (ufbxwte_deflate_impl)impl, level);
		result.min_delta[impl] = UINT64_MAX;
		result.max_delta[impl] = 0;
	}

	{
		FILE *f = fopen(file.data_path, "rb");
		size_t num_read = fread(g_src_buffer.data(), 1, file.data_size, f);
		ufbxwt_assert(num_read == file.data_size);
		fclose(f);
	}

	std::vector<size_t> uncompressed_offsets;

	ufbxwte_deflate_decompressor dc;
	ufbxwte_deflate_init_decompressor(&dc, UFBXWTE_DEFLATE_LIBDEFLATE);

	{
		size_t offset = 0;
		for (const test_array &arr : file.arrays) {
			result.uncompressed_bytes += arr.uncompressed_size;

			const void *src = g_src_buffer.data() + arr.offset;
			size_t src_size = arr.compressed_size;
			size_t result = ufbxwte_deflate_decompress(&dc, g_dst_buffer.data() + offset, arr.uncompressed_size, src, src_size);
			assert(result == arr.uncompressed_size);
			uncompressed_offsets.push_back(offset);
			offset += arr.uncompressed_size;
		}
	}

	for (size_t round = 0; round < 3; round++) {
		for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
			size_t total_size = 0;
			const uint64_t pre = cputime_cpu_tick();

			size_t index = 0;
			for (const test_array &arr : file.arrays) {
				void *dst = g_compress_buffer.data();
				size_t dst_size = g_compress_buffer.size();

				const void *src = g_dst_buffer.data() + uncompressed_offsets[index];
				size_t src_size = arr.uncompressed_size;

				size_t size = ufbxwte_deflate_compress(&cc[impl], dst, dst_size, src, src_size);
				if (size != SIZE_MAX) {
					total_size += size;
				}

				if (round == 0) {
					size_t dec_size = ufbxwte_deflate_decompress(&dc, g_check_buffer.data(), g_check_buffer.size(), dst, size);
					ufbxwt_assert(dec_size == arr.uncompressed_size);
				}

				index++;
			}
			const uint64_t post = cputime_cpu_tick();
			const uint64_t delta = post - pre;

			if (delta < result.min_delta[impl]) {
				result.min_delta[impl] = delta;
			}
			if (delta < result.max_delta[impl]) {
				result.max_delta[impl] = delta;
			}

			result.compressed_bytes[impl] = total_size;
		}
	}

	for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
		ufbxwte_deflate_free_compressor(&cc[impl]);
	}

	ufbxwte_deflate_free_decompressor(&dc);

	return result;
}

int main(int argc, char **argv)
{
	const char *index_path = nullptr;
	bool do_decompress = false;
	bool do_compress = false;
	int32_t compression_level = 6;

	im_arg_begin_c(argc, argv);
	while (im_arg_next()) {
		im_arg_help("--help", "show this help");

		if (im_arg("--verbose", "print verbose output")) {
			g_verbose = true;
		}
		if (im_arg("--decompress", "benchmark decompression")) {
			do_decompress = true;
		}
		if (im_arg("--compress", "benchmark compression")) {
			do_compress = true;
		}
		if (im_arg("--level level", "compression level")) {
			compression_level = im_arg_int(0);
		}
		if (im_arg("--index path", "path to index.json")) {
			index_path = im_arg_str(0);
		}
	}

	cputime_begin_init();
	uint64_t pre_json = cputime_cpu_tick();

	jsi_value *root = jsi_parse_file(index_path, NULL);
	uint64_t post_json = cputime_cpu_tick();
	cputime_end_init();

	if (g_verbose) {
		double duration = cputime_cpu_delta_to_sec(NULL, post_json - pre_json);
		printf("loaded index in %.2fms\n", duration * 1e3);
	}

	std::vector<test_file> test_files;
	size_t max_file_size = 0;
	size_t max_uncompressed_size = 0;
	size_t max_sum_uncompressed_size = 0;

	for (const jsi_value &file : jsi_get(root->object, "data")->array) {
		test_file tf;
		tf.source_path = jsi_get(file.object, "source_path")->string;
		tf.data_path = jsi_get(file.object, "data_path")->string;
		tf.data_size = (size_t)jsi_get(file.object, "data_size")->number;

		if (tf.data_size > max_file_size) {
			max_file_size = tf.data_size;
		}

		size_t total_uncompressed = 0;

		for (const jsi_value &value : jsi_get(file.object, "arrays")->array) {
			jsi_obj *val = value.object;

			test_array arr = { };
			arr.type = jsi_get(val, "type")->string[0];
			arr.array_path = jsi_get(val, "path")->string;
			arr.uncompressed_size = (int32_t)jsi_get(val, "size_u")->number;
			arr.compressed_size = (int32_t)jsi_get(val, "size_z")->number;
			arr.offset = (int32_t)jsi_get(val, "offset")->number;

			total_uncompressed += arr.uncompressed_size;

			tf.arrays.push_back(arr);
		}

		if (total_uncompressed > max_uncompressed_size) {
			max_uncompressed_size = total_uncompressed;
		}

		test_files.push_back(std::move(tf));
	}

	g_src_buffer.resize(max_file_size);
	g_dst_buffer.resize(max_uncompressed_size);
	g_compress_buffer.resize(max_uncompressed_size * 4 / 3 + 256);
	g_check_buffer.resize(max_uncompressed_size);

	if (do_decompress) {
		for (const test_file &file : test_files) {
			benchmark_decompress_result result = benchmark_decompress(file);
			printf("\n%s\n", file.source_path);
			for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
				double time = (double)cputime_cpu_delta_to_sec(NULL, result.min_delta[impl]);
				printf("%8s: %.2fms (%.2fns/b)\n", deflate_impl_name[impl], time * 1e3, time * 1e9 / (double)result.uncompressed_bytes);
			}
		}
	}

	if (do_compress) {
		for (const test_file &file : test_files) {
			benchmark_compress_result result = benchmark_compress(file, compression_level);
			printf("\n%s\n", file.source_path);
			for (size_t impl = 0; impl < UFBXWTE_DEFLATE_IMPL_COUNT; impl++) {
				double time = (double)cputime_cpu_delta_to_sec(NULL, result.min_delta[impl]);
				double size = (double)result.compressed_bytes[impl] / (double)result.uncompressed_bytes;
				printf("%8s: %.2fms, %.2f%%\n", deflate_impl_name[impl], time * 1e3, size * 100.0);
			}
		}
	}

	jsi_free(root);

	return 0;
}
