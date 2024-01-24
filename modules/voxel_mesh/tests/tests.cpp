/**************************************************************************/
/*  tests.cpp                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tests.h"
#include "../storage/voxel_data_map.h"
#include "../util/island_finder.h"
#include "../util/math/box3i.h"

#include <core/string/print_string.h>
#include <core/templates/hash_map.h>

void test_voxel_data_map_paste_fill() {
	static const int voxel_value = 1;
	static const int default_value = 0;
	static const int channel = VoxelBuffer::CHANNEL_TYPE;

	Ref<VoxelBuffer> buffer;
	buffer.instantiate();
	buffer->create(32, 16, 32);
	buffer->fill(voxel_value, channel);

	VoxelDataMap map;
	map.create(4, 0);

	const Box3i box(VoxelVector3i(10, 10, 10), buffer->get_size());

	map.paste(box.pos, **buffer, (1 << channel),
			std::numeric_limits<uint64_t>::max(), true);

	// All voxels in the area must be as pasted
	const bool is_match = box.all_cells_match([&map](const VoxelVector3i &pos) {
		return map.get_voxel(pos, channel) == voxel_value;
	});

	ERR_FAIL_COND(!is_match);

	// Check neighbor voxels to make sure they were not changed
	const Box3i padded_box = box.padded(1);
	bool outside_is_ok = true;
	padded_box.for_inner_outline(
			[&map, &outside_is_ok](const VoxelVector3i &pos) {
				if (map.get_voxel(pos, channel) != default_value) {
					outside_is_ok = false;
				}
			});

	ERR_FAIL_COND(!outside_is_ok);
}

void test_voxel_data_map_paste_mask() {
	static const int voxel_value = 1;
	static const int masked_value = 2;
	static const int default_value = 0;
	static const int channel = VoxelBuffer::CHANNEL_TYPE;

	Ref<VoxelBuffer> buffer;
	buffer.instantiate();
	buffer->create(32, 16, 32);
	// Fill the inside of the buffer with a value, and outline it with another
	// value, which we'll use as mask
	buffer->fill(masked_value, channel);
	for (int z = 1; z < buffer->get_size().z - 1; ++z) {
		for (int x = 1; x < buffer->get_size().x - 1; ++x) {
			for (int y = 1; y < buffer->get_size().y - 1; ++y) {
				buffer->set_voxel(voxel_value, x, y, z, channel);
			}
		}
	}

	VoxelDataMap map;
	map.create(4, 0);

	const Box3i box(VoxelVector3i(10, 10, 10), buffer->get_size());

	map.paste(box.pos, **buffer, (1 << channel), masked_value, true);

	// All voxels in the area must be as pasted. Ignoring the outline.
	const bool is_match =
			box.padded(-1).all_cells_match([&map](const VoxelVector3i &pos) {
				return map.get_voxel(pos, channel) == voxel_value;
			});

	/*for (int y = 0; y < buffer->get_size().y; ++y) {
			String line = String("y={0} | ").format(varray(y));
			for (int x = 0; x < buffer->get_size().x; ++x) {
					const int v = buffer->get_voxel(VoxelVector3i(x, y, box.pos.z
	+ 5), channel); if (v == default_value) { line += "- "; } else if (v ==
	voxel_value) { line += "O "; } else if (v == masked_value) { line += "M ";
					}
			}
			print_line(line);
	}

	for (int y = 0; y < 64; ++y) {
			String line = String("y={0} | ").format(varray(y));
			for (int x = 0; x < 64; ++x) {
					const int v = map.get_voxel(VoxelVector3i(x, y, box.pos.z +
	5), channel); if (v == default_value) { line += "- "; } else if (v ==
	voxel_value) { line += "O "; } else if (v == masked_value) { line += "M ";
					}
			}
			print_line(line);
	}*/

	ERR_FAIL_COND(!is_match);

	// Now check the outline voxels, they should be the same as before
	bool outside_is_ok = true;
	box.for_inner_outline([&map, &outside_is_ok](const VoxelVector3i &pos) {
		if (map.get_voxel(pos, channel) != default_value) {
			outside_is_ok = false;
		}
	});

	ERR_FAIL_COND(!outside_is_ok);
}

void test_voxel_data_map_copy() {
	static const int voxel_value = 1;
	static const int default_value = 0;
	static const int channel = VoxelBuffer::CHANNEL_TYPE;

	VoxelDataMap map;
	map.create(4, 0);

	Box3i box(10, 10, 10, 32, 16, 32);
	Ref<VoxelBuffer> buffer;
	buffer.instantiate();
	buffer->create(box.size);

	// Fill the inside of the buffer with a value, and leave outline to zero,
	// so our buffer isn't just uniform
	for (int z = 1; z < buffer->get_size().z - 1; ++z) {
		for (int x = 1; x < buffer->get_size().x - 1; ++x) {
			for (int y = 1; y < buffer->get_size().y - 1; ++y) {
				buffer->set_voxel(voxel_value, x, y, z, channel);
			}
		}
	}

	map.paste(box.pos, **buffer, (1 << channel), default_value, true);

	Ref<VoxelBuffer> buffer2;
	buffer2.instantiate();
	buffer2->create(box.size);

	map.copy(box.pos, **buffer2, (1 << channel));
	ERR_FAIL_COND(!buffer->equals(**buffer2));
}

void test_encode_weights_packed_u16() {
	FixedArray<uint8_t, 4> weights;
	// There is data loss of the 4 smaller bits in this encoding,
	// so to test this we may use values greater than 16.
	// There is a compromise in decoding, where we choose that only values
	// multiple of 16 are bijective.
	weights[0] = 1 << 4;
	weights[1] = 5 << 4;
	weights[2] = 10 << 4;
	weights[3] = 15 << 4;
	const uint16_t encoded_weights = encode_weights_to_packed_u16(
			weights[0], weights[1], weights[2], weights[3]);
	FixedArray<uint8_t, 4> decoded_weights =
			decode_weights_from_packed_u16(encoded_weights);
	ERR_FAIL_COND(weights != decoded_weights);
}

void test_copy_3d_region_zxy() {
	std::vector<uint16_t> src;
	std::vector<uint16_t> dst;
	const VoxelVector3i src_size(8, 8, 8);
	const VoxelVector3i dst_size(3, 4, 5);
	src.resize(src_size.volume(), 0);
	dst.resize(src_size.volume(), 0);
	for (unsigned int i = 0; i < src.size(); ++i) {
		src[i] = i;
	}

	Span<const uint16_t> srcs = to_span_const(src);
	Span<uint16_t> dsts = to_span(dst);
	const VoxelVector3i dst_min(0, 0, 0);
	const VoxelVector3i src_min(2, 1, 0);
	const VoxelVector3i src_max(5, 4, 3);
	copy_3d_region_zxy(dsts, dst_size, dst_min, srcs, src_size, src_min, src_max);
	VoxelVector3i pos;
	for (pos.z = src_min.z; pos.z < src_max.z; ++pos.z) {
		for (pos.x = src_min.x; pos.x < src_max.x; ++pos.x) {
			for (pos.y = src_min.y; pos.y < src_max.y; ++pos.y) {
				const uint16_t srcv = srcs[pos.get_zxy_index(src_size)];
				const uint16_t dstv =
						dsts[(pos - src_min + dst_min).get_zxy_index(dst_size)];
				ERR_FAIL_COND(srcv != dstv);
			}
		}
	}
}

void test_unordered_remove_if() {
	struct L {
		static unsigned int count(const std::vector<int> &vec, int v) {
			unsigned int n = 0;
			for (size_t i = 0; i < vec.size(); ++i) {
				if (vec[i] == v) {
					++n;
				}
			}
			return n;
		}
	};
	// Remove one at beginning
	{
		std::vector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 0; });

		ERR_FAIL_COND(vec.size() != 3);
		ERR_FAIL_COND((L::count(vec, 0) == 0 && L::count(vec, 1) == 1 &&
							  L::count(vec, 2) == 1 && L::count(vec, 3) == 1) == false);
	}
	// Remove one in middle
	{
		std::vector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 2; });

		ERR_FAIL_COND(vec.size() != 3);
		ERR_FAIL_COND((L::count(vec, 0) == 1 && L::count(vec, 1) == 1 &&
							  L::count(vec, 2) == 0 && L::count(vec, 3) == 1) == false);
	}
	// Remove one at end
	{
		std::vector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 3; });

		ERR_FAIL_COND(vec.size() != 3);
		ERR_FAIL_COND((L::count(vec, 0) == 1 && L::count(vec, 1) == 1 &&
							  L::count(vec, 2) == 1 && L::count(vec, 3) == 0) == false);
	}
	// Remove multiple
	{
		std::vector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 1 || v == 2; });

		ERR_FAIL_COND(vec.size() != 2);
		ERR_FAIL_COND((L::count(vec, 0) == 1 && L::count(vec, 1) == 0 &&
							  L::count(vec, 2) == 0 && L::count(vec, 3) == 1) == false);
	}
	// Remove last
	{
		std::vector<int> vec;
		vec.push_back(0);

		unordered_remove_if(vec, [](int v) { return v == 0; });

		ERR_FAIL_COND(vec.size() != 0);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VOXEL_TEST(fname)                                     \
	print_line(String("Running {0}").format(varray(#fname))); \
	fname()

void run_voxel_tests() {
	print_line("------------ Voxel tests begin -------------");

	VOXEL_TEST(test_voxel_data_map_paste_fill);
	VOXEL_TEST(test_voxel_data_map_paste_mask);
	VOXEL_TEST(test_voxel_data_map_copy);
	VOXEL_TEST(test_encode_weights_packed_u16);
	VOXEL_TEST(test_copy_3d_region_zxy);
	VOXEL_TEST(test_unordered_remove_if);

	print_line("------------ Voxel tests end -------------");
}
