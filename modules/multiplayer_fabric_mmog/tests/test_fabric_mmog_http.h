/**************************************************************************/
/*  test_fabric_mmog_http.h                                               */
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

#include "core/crypto/crypto.h"
#include "core/os/os.h"
#include "tests/test_macros.h"

namespace TestFabricMMOGHttp {

// Opt-in network test. Set FABRIC_MMOG_NETWORK_TESTS=1 to run. Hits the real
// V-Sekai desync store and verifies that http_get_blocking can round-trip a
// known chunk end-to-end: GET the `.cacnk`, zstd-decompress, SHA-512/256,
// and confirm the digest matches the filename. The chunk is small (~64 KB
// compressed) so the test stays quick when enabled.
TEST_CASE("[FabricMMOG][Network] http_get_blocking fetches a real V-Sekai chunk") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_NETWORK_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_NETWORK_TESTS=1 to enable network tests.");
		return;
	}

	// Godot's `--test` entry point skips the main() crypto bootstrap, so
	// TLS clients would fail with "SSL module failed to initialize!"
	// unless we load the default CA bundle here. The mbedtls impl ERR_FAILs
	// when called twice, so suppress the print — a sibling network test
	// may already have loaded them.
	ERR_PRINT_OFF;
	Crypto::load_default_certificates(String());
	ERR_PRINT_ON;

	// Known chunk from V-Sekai/casync-v-sekai-game @ main, shard 0006.
	static const uint8_t kExpectedId[FabricMMOGAsset::CHUNK_ID_BYTES] = {
		0x00,
		0x06,
		0xfc,
		0xc5,
		0x5e,
		0xba,
		0xbc,
		0x25,
		0xfd,
		0xdb,
		0x54,
		0xa6,
		0xd2,
		0xf5,
		0x95,
		0xee,
		0xde,
		0xe3,
		0xfb,
		0x87,
		0x6e,
		0x73,
		0x7d,
		0x21,
		0x08,
		0x28,
		0xb6,
		0xb8,
		0x5d,
		0x81,
		0x36,
		0x75,
	};

	const String url = FabricMMOGAsset::build_chunk_url(
			FabricMMOGAsset::DEFAULT_STORE_URL, kExpectedId);

	Vector<uint8_t> compressed;
	String error;
	const Error result = FabricMMOGAsset::http_get_blocking(url, compressed, error);

	CHECK_MESSAGE(result == OK, error);
	REQUIRE(compressed.size() > 0);

	Vector<uint8_t> decompressed;
	const Error verify_result = FabricMMOGAsset::decompress_and_verify_chunk(
			compressed, kExpectedId, decompressed, error);
	CHECK_MESSAGE(verify_result == OK, error);
	CHECK(decompressed.size() > 0);
}

} // namespace TestFabricMMOGHttp
