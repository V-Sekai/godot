/**************************************************************************/
/*  test_fabric_mmog_script_key.h                                         */
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

namespace TestFabricMMOGScriptKey {

TEST_CASE("[FabricMMOG] parse_script_key_json decodes base64 key, iv, and ttl") {
	// Uro returns AES_KEY_BYTES (16) and AES_IV_BYTES (12) as base64
	// strings so the key triple survives JSON transport. The fixture
	// mirrors Uro.Keys's seed in thirdparty/uro/test/script_key_test.exs.
	// 16 bytes of 0xAA -> "qqqqqqqqqqqqqqqqqqqqqg=="
	// 12 bytes of 0xBB -> "u7u7u7u7u7u7u7u7"
	const String json =
			"{\"key\":\"qqqqqqqqqqqqqqqqqqqqqg==\","
			"\"iv\":\"u7u7u7u7u7u7u7u7\","
			"\"ttl\":86400}";

	PackedByteArray key;
	PackedByteArray iv;
	uint64_t ttl = 0;
	String error;
	const Error result = FabricMMOGAsset::parse_script_key_json(
			json, key, iv, ttl, error);

	CHECK_MESSAGE(result == OK, error);
	REQUIRE(key.size() == FabricMMOGAsset::AES_KEY_BYTES);
	REQUIRE(iv.size() == FabricMMOGAsset::AES_IV_BYTES);
	CHECK(ttl == 86400);
	for (int i = 0; i < FabricMMOGAsset::AES_KEY_BYTES; i++) {
		CHECK(key[i] == 0xAA);
	}
	for (int i = 0; i < FabricMMOGAsset::AES_IV_BYTES; i++) {
		CHECK(iv[i] == 0xBB);
	}
}

TEST_CASE("[FabricMMOG] parse_script_key_json rejects malformed JSON") {
	PackedByteArray key;
	PackedByteArray iv;
	uint64_t ttl = 0;
	String error;
	const Error result = FabricMMOGAsset::parse_script_key_json(
			"{not json", key, iv, ttl, error);
	CHECK(result != OK);
	CHECK(key.is_empty());
	CHECK(iv.is_empty());
	CHECK(ttl == 0);
	CHECK_FALSE(error.is_empty());
}

TEST_CASE("[FabricMMOG] parse_script_key_json rejects missing fields") {
	PackedByteArray key;
	PackedByteArray iv;
	uint64_t ttl = 0;
	String error;
	const Error result = FabricMMOGAsset::parse_script_key_json(
			"{\"key\":\"qqqqqqqqqqqqqqqqqqqqqg==\"}",
			key, iv, ttl, error);
	CHECK(result != OK);
	CHECK(key.is_empty());
	CHECK(iv.is_empty());
	CHECK(error.contains("key/iv/ttl"));
}

TEST_CASE("[FabricMMOG] parse_script_key_json rejects wrong key length") {
	// 8 bytes of 0xAA -> "qqqqqqqqqqo=" — too short for AES_KEY_BYTES.
	const String json =
			"{\"key\":\"qqqqqqqqqqo=\","
			"\"iv\":\"u7u7u7u7u7u7u7u7\","
			"\"ttl\":86400}";
	PackedByteArray key;
	PackedByteArray iv;
	uint64_t ttl = 0;
	String error;
	const Error result = FabricMMOGAsset::parse_script_key_json(
			json, key, iv, ttl, error);
	CHECK(result != OK);
	CHECK(key.is_empty());
	CHECK(error.contains("length"));
}

TEST_CASE("[FabricMMOG] parse_script_key_json rejects wrong iv length") {
	// 8 bytes of 0xBB -> "u7u7u7u7u7s=" — too short for AES_IV_BYTES.
	const String json =
			"{\"key\":\"qqqqqqqqqqqqqqqqqqqqqg==\","
			"\"iv\":\"u7u7u7u7u7s=\","
			"\"ttl\":86400}";
	PackedByteArray key;
	PackedByteArray iv;
	uint64_t ttl = 0;
	String error;
	const Error result = FabricMMOGAsset::parse_script_key_json(
			json, key, iv, ttl, error);
	CHECK(result != OK);
	CHECK(key.is_empty());
	CHECK(iv.is_empty());
	CHECK(error.contains("length"));
}

TEST_CASE("[FabricMMOG] parse_script_key_json rejects non-base64 key") {
	const String json =
			"{\"key\":\"!!!notbase64!!!!\","
			"\"iv\":\"u7u7u7u7u7u7u7u7\","
			"\"ttl\":86400}";
	PackedByteArray key;
	PackedByteArray iv;
	uint64_t ttl = 0;
	String error;
	const Error result = FabricMMOGAsset::parse_script_key_json(
			json, key, iv, ttl, error);
	CHECK(result != OK);
	CHECK(key.is_empty());
}

// Opt-in network test. Set FABRIC_MMOG_NETWORK_TESTS=1 and stand up uro on
// 127.0.0.1:4000 with `Uro.Keys.put("asset:script-key-fixture", %{...})`
// seeded against the same AAA/BBB fixture the ExUnit suite uses.
TEST_CASE("[FabricMMOG][Network] request_asset_key round-trips against a local uro") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_NETWORK_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_NETWORK_TESTS=1 to enable network tests.");
		return;
	}
	ERR_PRINT_OFF;
	Crypto::load_default_certificates(String());
	ERR_PRINT_ON;

	PackedByteArray key;
	PackedByteArray iv;
	String error;
	const Error result = FabricMMOGAsset::request_asset_key(
			"http://127.0.0.1:4000", "asset:script-key-fixture",
			key, iv, error);
	CHECK_MESSAGE(result == OK, error);
	REQUIRE(key.size() == FabricMMOGAsset::AES_KEY_BYTES);
	REQUIRE(iv.size() == FabricMMOGAsset::AES_IV_BYTES);
	for (int i = 0; i < FabricMMOGAsset::AES_KEY_BYTES; i++) {
		CHECK(key[i] == 0xAA);
	}
	for (int i = 0; i < FabricMMOGAsset::AES_IV_BYTES; i++) {
		CHECK(iv[i] == 0xBB);
	}
}

} // namespace TestFabricMMOGScriptKey
