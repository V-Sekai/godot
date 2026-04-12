/**************************************************************************/
/*  test_fabric_mmog_assemble.h                                           */
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

#pragma once

#include "../fabric_mmog_asset.h"

#include "core/io/compression.h"
#include "tests/test_macros.h"

namespace TestFabricMMOGAssemble {

// Little-endian u64 append helper.
static void _push_u64(Vector<uint8_t> &p_buf, uint64_t p_value) {
	for (int i = 0; i < 8; i++) {
		p_buf.push_back(uint8_t((p_value >> (i * 8)) & 0xFF));
	}
}

static void _push_bytes(Vector<uint8_t> &p_buf, const uint8_t *p_src, int p_len) {
	for (int i = 0; i < p_len; i++) {
		p_buf.push_back(p_src[i]);
	}
}

// Build deterministic plain bytes for a chunk.
static Vector<uint8_t> _make_plain(int p_size, uint8_t p_seed) {
	Vector<uint8_t> out;
	out.resize(p_size);
	for (int i = 0; i < p_size; i++) {
		out.write[i] = uint8_t(i * 17 + p_seed * 53);
	}
	return out;
}

static Vector<uint8_t> _zstd_compress(const Vector<uint8_t> &p_plain) {
	const int64_t max_size = Compression::get_max_compressed_buffer_size(
			p_plain.size(), Compression::MODE_ZSTD);
	Vector<uint8_t> compressed;
	compressed.resize(max_size);
	const int64_t compressed_size = Compression::compress(
			compressed.ptrw(), p_plain.ptr(), p_plain.size(), Compression::MODE_ZSTD);
	compressed.resize(compressed_size);
	return compressed;
}

// Build a caibx buffer from a list of (offset_end, id) pairs.
static Vector<uint8_t> _build_caibx(const Vector<uint64_t> &p_ends,
		const Vector<Vector<uint8_t>> &p_ids) {
	Vector<uint8_t> buf;

	// FormatIndex — 48 bytes.
	_push_u64(buf, 48);
	_push_u64(buf, 0x96824d9c7b129ff9ULL);
	_push_u64(buf, 0xA000000000000000ULL);
	_push_u64(buf, 16 * 1024);
	_push_u64(buf, 64 * 1024);
	_push_u64(buf, 256 * 1024);

	// FormatTable header.
	_push_u64(buf, 0xFFFFFFFFFFFFFFFFULL);
	_push_u64(buf, 0xe75b9e112f17417dULL);

	for (int i = 0; i < p_ends.size(); i++) {
		_push_u64(buf, p_ends[i]);
		_push_bytes(buf, p_ids[i].ptr(), 32);
	}

	// Zero terminator + tail.
	_push_u64(buf, 0);
	_push_u64(buf, 0);
	_push_u64(buf, 48);
	_push_u64(buf, 16 + p_ends.size() * 40 + 40);
	_push_u64(buf, 0x4b4f050e5549ecd1ULL);

	return buf;
}

TEST_CASE("[FabricMMOG] assemble_from_caibx reassembles a two-chunk asset") {
	Vector<uint8_t> plain_a = _make_plain(100, 1);
	Vector<uint8_t> plain_b = _make_plain(200, 2);

	uint8_t id_a[FabricMMOGAsset::CHUNK_ID_BYTES];
	uint8_t id_b[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(plain_a.ptr(), plain_a.size(), id_a);
	FabricMMOGAsset::sha512_256(plain_b.ptr(), plain_b.size(), id_b);

	Vector<uint8_t> id_a_vec;
	id_a_vec.resize(32);
	memcpy(id_a_vec.ptrw(), id_a, 32);
	Vector<uint8_t> id_b_vec;
	id_b_vec.resize(32);
	memcpy(id_b_vec.ptrw(), id_b, 32);

	Vector<uint64_t> ends;
	ends.push_back(100);
	ends.push_back(300);
	Vector<Vector<uint8_t>> ids;
	ids.push_back(id_a_vec);
	ids.push_back(id_b_vec);

	const Vector<uint8_t> caibx = _build_caibx(ends, ids);

	HashMap<String, Vector<uint8_t>> chunks;
	chunks[FabricMMOGAsset::hex_from_id(id_a)] = _zstd_compress(plain_a);
	chunks[FabricMMOGAsset::hex_from_id(id_b)] = _zstd_compress(plain_b);

	Vector<uint8_t> output;
	String error;
	const Error result = FabricMMOGAsset::assemble_from_caibx(
			caibx, chunks, output, error);

	CHECK_MESSAGE(result == OK, error);
	REQUIRE(output.size() == 300);
	for (int i = 0; i < 100; i++) {
		CHECK(output[i] == plain_a[i]);
	}
	for (int i = 0; i < 200; i++) {
		CHECK(output[100 + i] == plain_b[i]);
	}
}

TEST_CASE("[FabricMMOG] assemble_from_caibx fails when a chunk is missing from the map") {
	Vector<uint8_t> plain_a = _make_plain(100, 1);
	Vector<uint8_t> plain_b = _make_plain(200, 2);

	uint8_t id_a[FabricMMOGAsset::CHUNK_ID_BYTES];
	uint8_t id_b[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(plain_a.ptr(), plain_a.size(), id_a);
	FabricMMOGAsset::sha512_256(plain_b.ptr(), plain_b.size(), id_b);

	Vector<uint8_t> id_a_vec;
	id_a_vec.resize(32);
	memcpy(id_a_vec.ptrw(), id_a, 32);
	Vector<uint8_t> id_b_vec;
	id_b_vec.resize(32);
	memcpy(id_b_vec.ptrw(), id_b, 32);

	Vector<uint64_t> ends;
	ends.push_back(100);
	ends.push_back(300);
	Vector<Vector<uint8_t>> ids;
	ids.push_back(id_a_vec);
	ids.push_back(id_b_vec);

	const Vector<uint8_t> caibx = _build_caibx(ends, ids);

	HashMap<String, Vector<uint8_t>> chunks;
	chunks[FabricMMOGAsset::hex_from_id(id_a)] = _zstd_compress(plain_a);
	// Deliberately omit chunk B.

	Vector<uint8_t> output;
	String error;
	const Error result = FabricMMOGAsset::assemble_from_caibx(
			caibx, chunks, output, error);

	CHECK(result != OK);
	CHECK_FALSE(error.is_empty());
}

TEST_CASE("[FabricMMOG] assemble_from_caibx rejects a chunk whose size disagrees with the index") {
	// Index claims chunk A is 100 bytes long, but the compressed blob in the
	// map decodes to 101 bytes. assemble_from_caibx must refuse rather than
	// silently drift the output layout.
	Vector<uint8_t> plain_real = _make_plain(101, 1);
	Vector<uint8_t> plain_b = _make_plain(200, 2);

	uint8_t id_real[FabricMMOGAsset::CHUNK_ID_BYTES];
	uint8_t id_b[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(plain_real.ptr(), plain_real.size(), id_real);
	FabricMMOGAsset::sha512_256(plain_b.ptr(), plain_b.size(), id_b);

	Vector<uint8_t> id_real_vec;
	id_real_vec.resize(32);
	memcpy(id_real_vec.ptrw(), id_real, 32);
	Vector<uint8_t> id_b_vec;
	id_b_vec.resize(32);
	memcpy(id_b_vec.ptrw(), id_b, 32);

	// Index says chunk A ends at 100 (size 100), but the map has the 101 B blob.
	Vector<uint64_t> ends;
	ends.push_back(100);
	ends.push_back(300);
	Vector<Vector<uint8_t>> ids;
	ids.push_back(id_real_vec);
	ids.push_back(id_b_vec);

	const Vector<uint8_t> caibx = _build_caibx(ends, ids);

	HashMap<String, Vector<uint8_t>> chunks;
	chunks[FabricMMOGAsset::hex_from_id(id_real)] = _zstd_compress(plain_real);
	chunks[FabricMMOGAsset::hex_from_id(id_b)] = _zstd_compress(plain_b);

	Vector<uint8_t> output;
	String error;
	const Error result = FabricMMOGAsset::assemble_from_caibx(
			caibx, chunks, output, error);

	CHECK(result != OK);
	CHECK_FALSE(error.is_empty());
}

} // namespace TestFabricMMOGAssemble
