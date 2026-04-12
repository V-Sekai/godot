/**************************************************************************/
/*  fabric_mmog_keystore.cpp                                              */
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

#include "fabric_mmog_keystore.h"

#include "core/crypto/crypto_core.h"
#include "core/io/json.h"
#include "core/os/time.h"

#include <keychain/keychain.h>

#include <string>

static String _gd_from_std(const std::string &p_s) {
	return String::utf8(p_s.c_str(), int(p_s.size()));
}

static std::string _std_from_gd(const String &p_s) {
	const CharString utf8 = p_s.utf8();
	return std::string(utf8.get_data(), size_t(utf8.length()));
}

static String _b64_encode_bytes(const PackedByteArray &p_bytes) {
	return CryptoCore::b64_encode_str(p_bytes.ptr(), size_t(p_bytes.size()));
}

static Error _b64_decode_bytes(const String &p_b64, int p_expected_len,
		PackedByteArray &r_out, String &r_error) {
	const CharString utf8 = p_b64.utf8();
	r_out.resize(p_expected_len);
	size_t decoded_len = 0;
	const Error err = CryptoCore::b64_decode(r_out.ptrw(), r_out.size(),
			&decoded_len,
			reinterpret_cast<const uint8_t *>(utf8.get_data()),
			utf8.length());
	if (err != OK) {
		r_error = "stored key material is not valid base64";
		r_out.resize(0);
		return ERR_INVALID_DATA;
	}
	if (int(decoded_len) != p_expected_len) {
		r_error = vformat("stored key material length %d != %d",
				int(decoded_len), p_expected_len);
		r_out.resize(0);
		return ERR_INVALID_DATA;
	}
	return OK;
}

static Error _kc_to_godot(const keychain::Error &p_err, String &r_error) {
	switch (p_err.type) {
		case keychain::ErrorType::NoError:
			return OK;
		case keychain::ErrorType::NotFound:
			r_error = "key material not found in OS key store";
			return ERR_FILE_NOT_FOUND;
		case keychain::ErrorType::AccessDenied:
			r_error = vformat("OS key store denied access: %s",
					_gd_from_std(p_err.message));
			return ERR_UNAUTHORIZED;
		case keychain::ErrorType::PasswordTooLong:
			r_error = "OS key store rejected blob: password too long";
			return ERR_INVALID_DATA;
		case keychain::ErrorType::GenericError:
		default:
			r_error = vformat("OS key store error: %s",
					_gd_from_std(p_err.message));
			return FAILED;
	}
}

Error FabricMMOGKeyStore::put(const String &p_asset_uuid,
		const PackedByteArray &p_key,
		const PackedByteArray &p_iv,
		String &r_error) {
	r_error = String();

	Dictionary blob;
	blob["key"] = _b64_encode_bytes(p_key);
	blob["iv"] = _b64_encode_bytes(p_iv);
	blob["stored_at"] = int64_t(Time::get_singleton()->get_unix_time_from_system());
	const String blob_json = JSON::stringify(blob);

	keychain::Error kc_err;
	keychain::setPassword(
			std::string(PACKAGE),
			std::string(SERVICE),
			_std_from_gd(p_asset_uuid),
			_std_from_gd(blob_json),
			kc_err);
	return _kc_to_godot(kc_err, r_error);
}

Error FabricMMOGKeyStore::get(const String &p_asset_uuid,
		PackedByteArray &r_key, PackedByteArray &r_iv,
		String &r_error) {
	return get_with_clock(p_asset_uuid,
			int64_t(Time::get_singleton()->get_unix_time_from_system()),
			r_key, r_iv, r_error);
}

Error FabricMMOGKeyStore::get_with_clock(const String &p_asset_uuid,
		int64_t p_now_unix,
		PackedByteArray &r_key, PackedByteArray &r_iv,
		String &r_error) {
	r_key.resize(0);
	r_iv.resize(0);
	r_error = String();

	keychain::Error kc_err;
	const std::string blob_std = keychain::getPassword(
			std::string(PACKAGE),
			std::string(SERVICE),
			_std_from_gd(p_asset_uuid),
			kc_err);
	const Error map_err = _kc_to_godot(kc_err, r_error);
	if (map_err != OK) {
		return map_err;
	}
	const String blob_json = _gd_from_std(blob_std);

	JSON json;
	const Error parse_err = json.parse(blob_json);
	if (parse_err != OK) {
		r_error = vformat("stored key material JSON parse error: %s",
				json.get_error_message());
		return ERR_INVALID_DATA;
	}
	const Variant parsed = json.get_data();
	if (parsed.get_type() != Variant::DICTIONARY) {
		r_error = "stored key material blob is not an object";
		return ERR_INVALID_DATA;
	}
	const Dictionary blob = parsed;
	if (!blob.has("key") || !blob.has("iv") || !blob.has("stored_at")) {
		r_error = "stored key material missing key/iv/stored_at";
		return ERR_INVALID_DATA;
	}

	const int64_t stored_at = int64_t(blob["stored_at"]);
	if (p_now_unix - stored_at > int64_t(FabricMMOGKeyStore::KEY_TTL_SECONDS)) {
		r_error = "expired";
		return ERR_FILE_CANT_READ;
	}

	const String key_b64 = blob["key"];
	if (_b64_decode_bytes(key_b64, FabricMMOGKeyStore::AES_KEY_BYTES, r_key, r_error) != OK) {
		return ERR_INVALID_DATA;
	}
	const String iv_b64 = blob["iv"];
	if (_b64_decode_bytes(iv_b64, FabricMMOGKeyStore::AES_IV_BYTES, r_iv, r_error) != OK) {
		r_key.resize(0);
		return ERR_INVALID_DATA;
	}
	return OK;
}

Error FabricMMOGKeyStore::remove(const String &p_asset_uuid, String &r_error) {
	r_error = String();

	keychain::Error kc_err;
	keychain::deletePassword(
			std::string(PACKAGE),
			std::string(SERVICE),
			_std_from_gd(p_asset_uuid),
			kc_err);
	if (kc_err.type == keychain::ErrorType::NotFound) {
		return OK;
	}
	return _kc_to_godot(kc_err, r_error);
}
