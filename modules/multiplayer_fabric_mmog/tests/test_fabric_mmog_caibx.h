/**************************************************************************/
/*  test_fabric_mmog_caibx.h                                              */
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

#include "tests/test_macros.h"

namespace TestFabricMMOGCaibx {

// Little-endian u64 append helper for building synthetic index buffers.
static void _push_u64(Vector<uint8_t> &p_buf, uint64_t p_value) {
	for (int i = 0; i < 8; i++) {
		p_buf.push_back(uint8_t((p_value >> (i * 8)) & 0xFF));
	}
}

static void _push_id(Vector<uint8_t> &p_buf, uint8_t p_fill) {
	for (int i = 0; i < 32; i++) {
		p_buf.push_back(p_fill);
	}
}

// Build a valid two-chunk caibx buffer matching the casync wire format:
// FormatIndex(48) + FormatTable(MAX_UINT64) + 2 items + 0 terminator
// + zero fill + index_offset + table_size + tail marker.
static Vector<uint8_t> _build_two_chunk_caibx() {
	Vector<uint8_t> buf;

	// FormatIndex — 48 bytes total.
	_push_u64(buf, 48); // size
	_push_u64(buf, 0x96824d9c7b129ff9ULL); // CaFormatIndex
	_push_u64(buf, 0xA000000000000000ULL); // CaFormatExcludeNoDump | CaFormatSHA512256
	_push_u64(buf, 16 * 1024); // min
	_push_u64(buf, 64 * 1024); // avg
	_push_u64(buf, 256 * 1024); // max

	// FormatTable header — size = MAX_UINT64, type = CaFormatTable.
	_push_u64(buf, 0xFFFFFFFFFFFFFFFFULL);
	_push_u64(buf, 0xe75b9e112f17417dULL);

	// Item 0: offset=100, id=all 0xAA.
	_push_u64(buf, 100);
	_push_id(buf, 0xAA);

	// Item 1: offset=300, id=all 0xBB.
	_push_u64(buf, 300);
	_push_id(buf, 0xBB);

	// Zero offset terminates the item loop.
	_push_u64(buf, 0);

	// Tail: zero fill 2, index offset, table size, tail marker.
	_push_u64(buf, 0);
	_push_u64(buf, 48);
	_push_u64(buf, 16 + 2 * 40 + 40); // arbitrary; decoder ignores
	_push_u64(buf, 0x4b4f050e5549ecd1ULL); // CaFormatTableTailMarker

	return buf;
}

TEST_CASE("[FabricMMOG] parse_caibx decodes a well-formed two-chunk index") {
	const Vector<uint8_t> buf = _build_two_chunk_caibx();

	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_caibx(buf, chunks, error);

	CHECK_MESSAGE(result == OK, error);
	REQUIRE(chunks.size() == 2);

	CHECK(chunks[0].start == 0);
	CHECK(chunks[0].size == 100);
	for (int i = 0; i < 32; i++) {
		CHECK(chunks[0].id[i] == 0xAA);
	}

	CHECK(chunks[1].start == 100);
	CHECK(chunks[1].size == 200);
	for (int i = 0; i < 32; i++) {
		CHECK(chunks[1].id[i] == 0xBB);
	}
}

TEST_CASE("[FabricMMOG] parse_caibx rejects a truncated buffer") {
	Vector<uint8_t> buf = _build_two_chunk_caibx();
	buf.resize(buf.size() - 8); // Chop off the tail marker.

	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_caibx(buf, chunks, error);

	CHECK(result != OK);
	CHECK_FALSE(error.is_empty());
}

TEST_CASE("[FabricMMOG] parse_caibx rejects the wrong magic type") {
	Vector<uint8_t> buf = _build_two_chunk_caibx();
	// Corrupt the FormatIndex type field (bytes 8..15).
	buf.write[8] = 0x00;

	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_caibx(buf, chunks, error);

	CHECK(result != OK);
	CHECK_FALSE(error.is_empty());
}

} // namespace TestFabricMMOGCaibx
