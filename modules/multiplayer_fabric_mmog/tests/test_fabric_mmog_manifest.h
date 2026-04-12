/**************************************************************************/
/*  test_fabric_mmog_manifest.h                                           */
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

namespace TestFabricMMOGManifest {

TEST_CASE("[FabricMMOG] parse_manifest_json decodes a two-chunk response") {
	// The uro /storage/:id/manifest endpoint returns chunks keyed by the
	// same SHA-512/256 address used inside a caibx. Each chunk id is the
	// 64-character lowercase hex digest; start/size are u64.
	String id_a;
	String id_b;
	for (int i = 0; i < FabricMMOGAsset::CHUNK_ID_BYTES * 2; i++) {
		id_a += "a";
		id_b += "b";
	}

	const String json =
			"{\"chunks\":[" +
			String("{\"id\":\"") + id_a + "\",\"start\":0,\"size\":1024},"
										  "{\"id\":\"" +
			id_b + "\",\"start\":1024,\"size\":2048}]}";

	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_manifest_json(json, chunks, error);

	CHECK_MESSAGE(result == OK, error);
	REQUIRE(chunks.size() == 2);
	CHECK(chunks[0].start == 0);
	CHECK(chunks[0].size == 1024);
	CHECK(chunks[1].start == 1024);
	CHECK(chunks[1].size == 2048);

	// First byte of id_a hex-decodes to 0xaa, id_b to 0xbb.
	CHECK(chunks[0].id[0] == 0xaa);
	CHECK(chunks[0].id[FabricMMOGAsset::CHUNK_ID_BYTES - 1] == 0xaa);
	CHECK(chunks[1].id[0] == 0xbb);
	CHECK(chunks[1].id[FabricMMOGAsset::CHUNK_ID_BYTES - 1] == 0xbb);
}

TEST_CASE("[FabricMMOG] parse_manifest_json rejects malformed JSON") {
	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_manifest_json(
			"{not json", chunks, error);
	CHECK(result != OK);
	CHECK(chunks.is_empty());
	CHECK_FALSE(error.is_empty());
}

TEST_CASE("[FabricMMOG] parse_manifest_json rejects non-array chunks") {
	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_manifest_json(
			"{\"chunks\":\"nope\"}", chunks, error);
	CHECK(result != OK);
	CHECK(chunks.is_empty());
	CHECK(error.contains("chunks"));
}

TEST_CASE("[FabricMMOG] parse_manifest_json rejects wrong-length chunk id") {
	const String json =
			"{\"chunks\":[{\"id\":\"deadbeef\",\"start\":0,\"size\":1}]}";
	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_manifest_json(json, chunks, error);
	CHECK(result != OK);
	CHECK(chunks.is_empty());
	CHECK(error.contains("length"));
}

TEST_CASE("[FabricMMOG] parse_manifest_json rejects non-hex chunk id") {
	String id;
	for (int i = 0; i < FabricMMOGAsset::CHUNK_ID_BYTES * 2; i++) {
		id += "z";
	}
	const String json =
			String("{\"chunks\":[{\"id\":\"") + id +
			"\",\"start\":0,\"size\":1}]}";
	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::parse_manifest_json(json, chunks, error);
	CHECK(result != OK);
	CHECK(chunks.is_empty());
	CHECK(error.contains("hex"));
}

// Opt-in network test. Set FABRIC_MMOG_NETWORK_TESTS=1 and stand up uro on
// 127.0.0.1:4000 with `Uro.Manifest.put("asset:fixture", [...])` seeded
// against the same two-chunk fixture the ExUnit suite uses. Verifies that
// `request_manifest` can POST and decode in one round trip.
TEST_CASE("[FabricMMOG][Network] request_manifest round-trips against a local uro") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_NETWORK_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_NETWORK_TESTS=1 to enable network tests.");
		return;
	}
	ERR_PRINT_OFF;
	Crypto::load_default_certificates(String());
	ERR_PRINT_ON;

	Vector<FabricMMOGAsset::CaibxChunk> chunks;
	String error;
	const Error result = FabricMMOGAsset::request_manifest(
			"http://127.0.0.1:4000", "asset:fixture", chunks, error);
	CHECK_MESSAGE(result == OK, error);
	REQUIRE(chunks.size() == 2);
	CHECK(chunks[0].start == 0);
	CHECK(chunks[0].size == 1024);
	CHECK(chunks[1].start == 1024);
	CHECK(chunks[1].size == 2048);
}

} // namespace TestFabricMMOGManifest
