/**************************************************************************/
/*  fabric_mmog_keystore.h                                                */
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

#include "core/error/error_list.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

// Thin wrapper around hrantzsch/keychain for persisting per-asset AES key
// material in OS secure storage (macOS Keychain, Windows Credential Vault,
// Linux libsecret). Isolates the <keychain/keychain.h> include — which
// pulls in Security.framework / wincred / libsecret headers — from the
// rest of the module.
class FabricMMOGKeyStore {
public:
	static constexpr int AES_KEY_BYTES = 16; // AES-128
	static constexpr int AES_IV_BYTES = 12; // GCM standard
	static constexpr int KEY_TTL_SECONDS = 86400; // 24 h

	// Reverse-DNS package identifier used for all stored entries. The
	// SERVICE string namespaces this module's secrets inside that package.
	static constexpr const char *PACKAGE = "org.v-sekai.godot";
	static constexpr const char *SERVICE = "multiplayer_fabric_mmog.asset_key";

	// Persist raw `p_key` (AES_KEY_BYTES) + `p_iv` (AES_IV_BYTES) under
	// `p_asset_uuid`. Overwrites any existing entry. The blob is serialized
	// as `{"key":"<b64>","iv":"<b64>","stored_at":<unix>}` so TTL can be
	// enforced on read without a second OS call.
	static Error put(const String &p_asset_uuid,
			const PackedByteArray &p_key,
			const PackedByteArray &p_iv,
			String &r_error);

	// Look up `p_asset_uuid` and return its key + iv. Returns ERR_FILE_NOT_FOUND
	// when no entry exists, and ERR_FILE_CANT_READ (with `r_error` set to
	// "expired") when the stored blob is older than KEY_TTL_SECONDS.
	static Error get(const String &p_asset_uuid,
			PackedByteArray &r_key, PackedByteArray &r_iv,
			String &r_error);

	// Injected-clock variant used by tests. `p_now_unix` is the "current"
	// unix time in seconds; TTL is enforced against the stored blob's
	// `stored_at` field. The no-clock `get` overload calls this with
	// `Time::get_unix_time_from_system()`.
	static Error get_with_clock(const String &p_asset_uuid,
			int64_t p_now_unix,
			PackedByteArray &r_key, PackedByteArray &r_iv,
			String &r_error);

	// Remove `p_asset_uuid`'s entry. Deleting a missing entry is a no-op
	// (returns OK) — callers don't need to check existence first.
	static Error remove(const String &p_asset_uuid, String &r_error);
};
