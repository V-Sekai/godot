/**************************************************************************/
/*  test_fabric_mmog_verify.h                                             */
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

#include <zstd.h>

namespace TestFabricMMOGVerify {

static String _hex(const uint8_t *p_data, int p_len) {
	static const char kHex[] = "0123456789abcdef";
	String out;
	for (int i = 0; i < p_len; i++) {
		char pair[3] = { kHex[(p_data[i] >> 4) & 0x0F], kHex[p_data[i] & 0x0F], 0 };
		out += pair;
	}
	return out;
}

TEST_CASE("[FabricMMOG] sha512_256 matches the NIST empty-string vector") {
	uint8_t digest[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(nullptr, 0, digest);
	CHECK(_hex(digest, FabricMMOGAsset::CHUNK_ID_BYTES) ==
			"c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a");
}

TEST_CASE("[FabricMMOG] sha512_256 matches the NIST \"abc\" vector") {
	const char *msg = "abc";
	uint8_t digest[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(reinterpret_cast<const uint8_t *>(msg), 3, digest);
	CHECK(_hex(digest, FabricMMOGAsset::CHUNK_ID_BYTES) ==
			"53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23");
}

TEST_CASE("[FabricMMOG] sha512_256 matches the NIST fox vector") {
	const char *msg = "The quick brown fox jumps over the lazy dog";
	uint8_t digest[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(reinterpret_cast<const uint8_t *>(msg),
			strlen(msg), digest);
	CHECK(_hex(digest, FabricMMOGAsset::CHUNK_ID_BYTES) ==
			"dd9d67b371519c339ed8dbd25af90e976a1eeefd4ad3d889005e532fc5bef04d");
}

TEST_CASE("[FabricMMOG] decompress_and_verify_chunk round-trips a zstd blob") {
	// Build plain bytes large enough to exercise zstd meaningfully.
	Vector<uint8_t> plain;
	plain.resize(4096);
	for (int i = 0; i < plain.size(); i++) {
		plain.write[i] = uint8_t(i * 31 + 7);
	}

	// Expected chunk ID = SHA-512/256 of the plain bytes.
	uint8_t expected_id[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(plain.ptr(), plain.size(), expected_id);

	// Compress with Godot's zstd wrapper.
	const int64_t max_size = Compression::get_max_compressed_buffer_size(
			plain.size(), Compression::MODE_ZSTD);
	Vector<uint8_t> compressed;
	compressed.resize(max_size);
	const int64_t compressed_size = Compression::compress(
			compressed.ptrw(), plain.ptr(), plain.size(), Compression::MODE_ZSTD);
	REQUIRE(compressed_size > 0);
	compressed.resize(compressed_size);

	Vector<uint8_t> decompressed;
	String error;
	const Error result = FabricMMOGAsset::decompress_and_verify_chunk(
			compressed, expected_id, decompressed, error);

	CHECK_MESSAGE(result == OK, error);
	REQUIRE(decompressed.size() == plain.size());
	for (int i = 0; i < plain.size(); i++) {
		CHECK(decompressed[i] == plain[i]);
	}
}

TEST_CASE("[FabricMMOG] decompress_and_verify_chunk handles a zstd frame without a declared content size") {
	// Real casync/desync chunks are produced by the Go `desync` tool, which
	// writes zstd frames with the content-size flag cleared. Godot's
	// `Compression::compress` wrapper always sets the flag, so reproduce the
	// desync shape here directly via libzstd with contentSizeFlag=0.
	Vector<uint8_t> plain;
	plain.resize(2048);
	for (int i = 0; i < plain.size(); i++) {
		plain.write[i] = uint8_t((i * 11 + 3) ^ 0x5A);
	}

	uint8_t expected_id[FabricMMOGAsset::CHUNK_ID_BYTES];
	FabricMMOGAsset::sha512_256(plain.ptr(), plain.size(), expected_id);

	ZSTD_CCtx *cctx = ZSTD_createCCtx();
	REQUIRE(cctx != nullptr);
	REQUIRE(!ZSTD_isError(ZSTD_CCtx_setParameter(cctx, ZSTD_c_contentSizeFlag, 0)));

	const size_t bound = ZSTD_compressBound(plain.size());
	Vector<uint8_t> compressed;
	compressed.resize(int(bound));
	const size_t compressed_size = ZSTD_compress2(
			cctx, compressed.ptrw(), bound, plain.ptr(), plain.size());
	REQUIRE(!ZSTD_isError(compressed_size));
	ZSTD_freeCCtx(cctx);
	compressed.resize(int(compressed_size));

	// Sanity: confirm the produced frame really has no declared content size.
	const unsigned long long declared = ZSTD_getFrameContentSize(
			compressed.ptr(), compressed.size());
	REQUIRE(declared == ZSTD_CONTENTSIZE_UNKNOWN);

	Vector<uint8_t> decompressed;
	String error;
	const Error result = FabricMMOGAsset::decompress_and_verify_chunk(
			compressed, expected_id, decompressed, error);

	CHECK_MESSAGE(result == OK, error);
	REQUIRE(decompressed.size() == plain.size());
	for (int i = 0; i < plain.size(); i++) {
		CHECK(decompressed[i] == plain[i]);
	}
}

TEST_CASE("[FabricMMOG] decompress_and_verify_chunk rejects a mismatched chunk ID") {
	Vector<uint8_t> plain;
	plain.resize(128);
	for (int i = 0; i < plain.size(); i++) {
		plain.write[i] = uint8_t(i);
	}

	const int64_t max_size = Compression::get_max_compressed_buffer_size(
			plain.size(), Compression::MODE_ZSTD);
	Vector<uint8_t> compressed;
	compressed.resize(max_size);
	const int64_t compressed_size = Compression::compress(
			compressed.ptrw(), plain.ptr(), plain.size(), Compression::MODE_ZSTD);
	REQUIRE(compressed_size > 0);
	compressed.resize(compressed_size);

	uint8_t wrong_id[FabricMMOGAsset::CHUNK_ID_BYTES] = {};
	wrong_id[0] = 0xDE;
	wrong_id[1] = 0xAD;

	Vector<uint8_t> decompressed;
	String error;
	const Error result = FabricMMOGAsset::decompress_and_verify_chunk(
			compressed, wrong_id, decompressed, error);

	CHECK(result != OK);
	CHECK_FALSE(error.is_empty());
}

TEST_CASE("[FabricMMOG] decompress_and_verify_chunk rejects a corrupt zstd frame") {
	Vector<uint8_t> garbage;
	garbage.resize(64);
	for (int i = 0; i < garbage.size(); i++) {
		garbage.write[i] = uint8_t(0xAB);
	}

	uint8_t id[FabricMMOGAsset::CHUNK_ID_BYTES] = {};

	Vector<uint8_t> decompressed;
	String error;
	const Error result = FabricMMOGAsset::decompress_and_verify_chunk(
			garbage, id, decompressed, error);

	CHECK(result != OK);
	CHECK_FALSE(error.is_empty());
}

} // namespace TestFabricMMOGVerify
