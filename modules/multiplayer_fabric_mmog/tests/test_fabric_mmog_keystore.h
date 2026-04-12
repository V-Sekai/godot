/**************************************************************************/
/*  test_fabric_mmog_keystore.h                                           */
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

#include "modules/modules_enabled.gen.h" // MODULE_KEYCHAIN_ENABLED

#ifdef MODULE_KEYCHAIN_ENABLED

#include "../fabric_mmog_asset.h"

#include "core/os/os.h"
#include "tests/test_macros.h"

#include "modules/keychain/fabric_mmog_keystore.h"

namespace TestFabricMMOGKeyStore {

// Opt-in OS-keychain round-trip. Set FABRIC_MMOG_KEYCHAIN_TESTS=1 to run;
// unconditional tests touch the real user keychain and would be surprising
// behavior in CI. The test uses a clearly-namespaced asset uuid and
// deletes its entry on the way in and the way out so repeated runs don't
// accumulate state.
TEST_CASE("[FabricMMOG][Keychain] put/get/remove round-trips through the OS key store") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_KEYCHAIN_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_KEYCHAIN_TESTS=1 to enable keychain tests.");
		return;
	}

	const String uuid = "asset:fabric-mmog-keystore-test";
	String error;

	// Clean up any leftover entry from a previous run.
	FabricMMOGKeyStore::remove(uuid, error);
	error = String();

	PackedByteArray key;
	key.resize(FabricMMOGAsset::AES_KEY_BYTES);
	for (int i = 0; i < key.size(); i++) {
		key.write[i] = uint8_t(0xCC);
	}
	PackedByteArray iv;
	iv.resize(FabricMMOGAsset::AES_IV_BYTES);
	for (int i = 0; i < iv.size(); i++) {
		iv.write[i] = uint8_t(0xDD);
	}

	const Error put_err = FabricMMOGKeyStore::put(uuid, key, iv, error);
	CHECK_MESSAGE(put_err == OK, error);

	PackedByteArray got_key;
	PackedByteArray got_iv;
	const Error get_err = FabricMMOGKeyStore::get(uuid, got_key, got_iv, error);
	CHECK_MESSAGE(get_err == OK, error);
	REQUIRE(got_key.size() == FabricMMOGAsset::AES_KEY_BYTES);
	REQUIRE(got_iv.size() == FabricMMOGAsset::AES_IV_BYTES);
	for (int i = 0; i < got_key.size(); i++) {
		CHECK(got_key[i] == 0xCC);
	}
	for (int i = 0; i < got_iv.size(); i++) {
		CHECK(got_iv[i] == 0xDD);
	}

	// Clean up so the user's keychain doesn't accumulate test entries.
	const Error rm_err = FabricMMOGKeyStore::remove(uuid, error);
	CHECK_MESSAGE(rm_err == OK, error);

	// After removal, get() should report not-found.
	PackedByteArray after_key;
	PackedByteArray after_iv;
	error = String();
	const Error missing_err = FabricMMOGKeyStore::get(uuid, after_key, after_iv, error);
	CHECK(missing_err == ERR_FILE_NOT_FOUND);
	CHECK(after_key.is_empty());
	CHECK(after_iv.is_empty());
}

TEST_CASE("[FabricMMOG][Keychain] get_with_clock enforces KEY_TTL_SECONDS") {
	if (OS::get_singleton()->get_environment("FABRIC_MMOG_KEYCHAIN_TESTS") != "1") {
		MESSAGE("Skipping: set FABRIC_MMOG_KEYCHAIN_TESTS=1 to enable keychain tests.");
		return;
	}

	const String uuid = "asset:fabric-mmog-keystore-ttl-test";
	String error;

	FabricMMOGKeyStore::remove(uuid, error);
	error = String();

	PackedByteArray key;
	key.resize(FabricMMOGAsset::AES_KEY_BYTES);
	for (int i = 0; i < key.size(); i++) {
		key.write[i] = uint8_t(0x11);
	}
	PackedByteArray iv;
	iv.resize(FabricMMOGAsset::AES_IV_BYTES);
	for (int i = 0; i < iv.size(); i++) {
		iv.write[i] = uint8_t(0x22);
	}

	const Error put_err = FabricMMOGKeyStore::put(uuid, key, iv, error);
	CHECK_MESSAGE(put_err == OK, error);

	const int64_t now = int64_t(OS::get_singleton()->get_unix_time());

	// Within TTL: still readable.
	PackedByteArray fresh_key;
	PackedByteArray fresh_iv;
	error = String();
	const Error fresh_err = FabricMMOGKeyStore::get_with_clock(uuid, now,
			fresh_key, fresh_iv, error);
	CHECK_MESSAGE(fresh_err == OK, error);
	CHECK(fresh_key.size() == FabricMMOGAsset::AES_KEY_BYTES);
	CHECK(fresh_iv.size() == FabricMMOGAsset::AES_IV_BYTES);

	// Past TTL: expired.
	PackedByteArray stale_key;
	PackedByteArray stale_iv;
	error = String();
	const int64_t far_future = now + int64_t(FabricMMOGAsset::KEY_TTL_SECONDS) + 1;
	const Error stale_err = FabricMMOGKeyStore::get_with_clock(uuid, far_future,
			stale_key, stale_iv, error);
	CHECK(stale_err == ERR_FILE_CANT_READ);
	CHECK(error == "expired");
	CHECK(stale_key.is_empty());
	CHECK(stale_iv.is_empty());

	FabricMMOGKeyStore::remove(uuid, error);
}

} // namespace TestFabricMMOGKeyStore

#endif // MODULE_KEYCHAIN_ENABLED
