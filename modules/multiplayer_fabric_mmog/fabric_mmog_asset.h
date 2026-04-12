/**************************************************************************/
/*  fabric_mmog_asset.h                                                   */
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

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"

// FabricMMOGAsset — uro + desync asset delivery wrapper.
//
// Thin bridge to the desync content-addressable store and the uro backend:
// fetch a chunk index, reassemble a .scn / .glb / .vrm / ELF, verify
// integrity, cache locally, hand the path back to Godot's ResourceLoader or
// godot-sandbox. All numeric parameters (chunk sizes, hash widths, key
// formats, TTLs) are constants on this class so the C++ header is the
// canonical reference.
class FabricMMOGAsset : public RefCounted {
	GDCLASS(FabricMMOGAsset, RefCounted);

public:
	// ── desync chunk parameters ─────────────────────────────────────────
	// Chunk IDs are SHA-512/256 (32-byte digest of the first 256 bits of a
	// SHA-512 hash). Matches the upstream folbricht/desync default.
	static constexpr int CHUNK_ID_BYTES = 32;
	// Variable chunk size window; desync picks cut points via a rolling
	// hash within these bounds.
	static constexpr int CHUNK_MIN_BYTES = 16 * 1024; // 16 KB
	static constexpr int CHUNK_MAX_BYTES = 256 * 1024; // 256 KB

	// ── Script / asset registry entry (sent on CH_MIGRATION at join) ────
	// Wire format: [slot: u16][index_chunk_id: 32B][uro_uuid: 16B]
	static constexpr int REGISTRY_SLOT_BYTES = 2;
	static constexpr int REGISTRY_INDEX_ID_BYTES = CHUNK_ID_BYTES; // 32
	static constexpr int REGISTRY_URO_UUID_BYTES = 16;
	static constexpr int REGISTRY_ENTRY_BYTES = REGISTRY_SLOT_BYTES +
			REGISTRY_INDEX_ID_BYTES + REGISTRY_URO_UUID_BYTES; // 50

	// ── Encryption (per-asset AES-128-GCM) ──────────────────────────────
	static constexpr int AES_KEY_BYTES = 16; // AES-128
	static constexpr int AES_IV_BYTES = 12; // GCM standard
	static constexpr int AES_TAG_BYTES = 16; // GCM auth tag
	static constexpr int KEY_TTL_SECONDS = 86400; // 24 h

	// ── uro endpoints ───────────────────────────────────────────────────
	// Relative paths; host is provided at init time.
	static constexpr const char *URO_PATH_SCRIPT_KEY = "/auth/script_key";
	static constexpr const char *URO_PATH_MANIFEST = "/storage/manifest";

	// Default public desync store URL used during development. Points at
	// raw.githubusercontent.com because the GitHub Pages mirror at
	// v-sekai.github.io returns 404 for individual chunk files — Pages only
	// serves the README.
	static constexpr const char *DEFAULT_STORE_URL =
			"https://raw.githubusercontent.com/V-Sekai/casync-v-sekai-game/main/store";

	// ── .caibx index parser ─────────────────────────────────────────────
	// One decoded chunk table entry. `start` and `size` are in the
	// reassembled (uncompressed) stream; `id` is the SHA-512/256 content
	// address used to pull the chunk from a desync store.
	struct CaibxChunk {
		uint8_t id[CHUNK_ID_BYTES];
		uint64_t start;
		uint64_t size;
	};

	// Parse a desync `.caibx` index file from a byte buffer. Follows the
	// casync wire format: a FormatIndex header (type 0x96824d9c7b129ff9,
	// size 48) followed by a FormatTable (type 0xe75b9e112f17417d, size
	// MAX_UINT64) of [offset u64][chunk_id 32B] items terminated by a zero
	// offset + tail marker 0x4b4f050e5549ecd1. Returns OK on success and
	// populates `r_chunks`; on failure sets `r_error` with a human-readable
	// reason.
	static Error parse_caibx(const Vector<uint8_t> &p_bytes,
			Vector<CaibxChunk> &r_chunks, String &r_error);

	// Build the desync HTTP store URL for a compressed chunk. desync shards
	// the store by the first 4 lowercase hex characters of the chunk ID, so
	// the layout is `{store}/xxxx/<full-hex>.cacnk`. Any trailing slash on
	// the store URL is stripped so both forms produce identical output.
	static String build_chunk_url(const String &p_store_url,
			const uint8_t p_chunk_id[CHUNK_ID_BYTES]);

	// Compute the SHA-512/256 digest (FIPS 180-4) of a byte buffer. This is
	// the chunk ID algorithm used by casync/desync — not a truncation of
	// SHA-512 — and Godot's HashingContext doesn't expose it, so we carry a
	// small standalone implementation here.
	static void sha512_256(const uint8_t *p_data, int64_t p_len,
			uint8_t r_digest[CHUNK_ID_BYTES]);

	// Decompress a `.cacnk` blob with zstd and verify that the SHA-512/256
	// of the decompressed bytes matches the expected chunk ID. Writes the
	// reassembled bytes to `r_decompressed` on success, or sets `r_error`
	// on failure (corrupt zstd frame, hash mismatch, etc.).
	static Error decompress_and_verify_chunk(
			const Vector<uint8_t> &p_compressed,
			const uint8_t p_expected_id[CHUNK_ID_BYTES],
			Vector<uint8_t> &r_decompressed, String &r_error);

	// Lowercase hex encoding of a chunk ID. Used as the key type for the
	// assemble step's chunk map, matching the filename stem under the
	// desync store's shard directories.
	static String hex_from_id(const uint8_t p_chunk_id[CHUNK_ID_BYTES]);

	// Reassemble an asset from a parsed `.caibx` index and a map of already
	// fetched+decompressed chunks keyed by lowercase hex chunk ID. Pure
	// logic — does no I/O — so the HTTP driver in `fetch_asset` can delegate
	// the reassembly step here and stay trivially testable. Fails if any
	// chunk is missing from the map or its size disagrees with the index.
	static Error assemble_from_caibx(
			const Vector<uint8_t> &p_caibx_bytes,
			const HashMap<String, Vector<uint8_t>> &p_chunks_by_hex,
			Vector<uint8_t> &r_output, String &r_error);

	// Blocking HTTP(S) GET that writes the response body to `r_body`.
	// Follows up to 5 redirects so the helper can sit behind short-link
	// hosts. Intended for one-shot chunk/index downloads from the desync
	// store; higher-level code dispatches it off the main thread when used
	// in real clients. On failure returns a non-OK error and populates
	// `r_error` with a human-readable reason.
	static Error http_get_blocking(const String &p_url,
			Vector<uint8_t> &r_body, String &r_error);

	// Generic blocking HTTP(S) request. `p_method` is `"GET"` or `"POST"`.
	// For POST, `p_body` is sent as the request body and a `Content-Type`
	// header from `p_content_type` (empty string means omit). Response body
	// is written to `r_body`. Shares the redirect-following logic with
	// `http_get_blocking`, which now delegates here.
	static Error http_request_blocking(const String &p_method,
			const String &p_url,
			const Vector<uint8_t> &p_body,
			const String &p_content_type,
			Vector<uint8_t> &r_body, String &r_error);

	// ── uro manifest + ACL ──────────────────────────────────────────────
	// Parse a manifest endpoint response into a list of chunk descriptors.
	// Wire format (application/json):
	//   {"chunks":[{"id":"<64-hex>","start":<u64>,"size":<u64>}, ...]}
	// `id` is the SHA-512/256 chunk address (CHUNK_ID_BYTES, hex-encoded).
	// Returns OK on success and populates `r_chunks`; on failure sets
	// `r_error` with a human-readable reason and leaves `r_chunks` empty.
	// Pure logic — no I/O — so it unit-tests without a live backend.
	static Error parse_manifest_json(const String &p_json,
			Vector<CaibxChunk> &r_chunks, String &r_error);

	// POST to uro's /acl/check endpoint to resolve a ReBAC tuple
	// `(p_object, p_relation, p_subject)` against the relation graph.
	// Returns OK and sets `r_allowed` on success, or a non-OK error and
	// populates `r_error` on transport/parse failure. Tuple components are
	// opaque strings — `"asset:123"`, `"viewer"`, `"user:456"` — matching
	// the ReBAC convention documented in CONCEPT_MMOG.md.
	static Error acl_check(const String &p_uro_base_url,
			const String &p_object,
			const String &p_relation,
			const String &p_subject,
			bool &r_allowed, String &r_error);

	// ── Fetch stubs ─────────────────────────────────────────────────────
	// Fetch-and-reassemble an asset by its index chunk ID. Returns the
	// local path on success, or an empty string on failure.
	String fetch_asset(const String &p_store_url,
			const String &p_index_url,
			const String &p_output_dir,
			const String &p_cache_dir);

	// POST to uro's /storage/:id/manifest endpoint and decode the chunk
	// list in one round trip. On success writes the parsed chunks to
	// `r_chunks`; on failure sets `r_error`.
	static Error request_manifest(const String &p_uro_base_url,
			const String &p_asset_id,
			Vector<CaibxChunk> &r_chunks, String &r_error);

	// Parse a `/auth/script_key` response into raw key + iv bytes.
	// Wire format (application/json):
	//   {"key":"<base64 AES_KEY_BYTES>",
	//    "iv":"<base64 AES_IV_BYTES>",
	//    "ttl":<u64 seconds>}
	// Returns OK on success and populates `r_key`, `r_iv`, `r_ttl`; on
	// failure sets `r_error` with a human-readable reason and leaves
	// the out-params empty. Pure logic — no I/O.
	static Error parse_script_key_json(const String &p_json,
			PackedByteArray &r_key, PackedByteArray &r_iv,
			uint64_t &r_ttl, String &r_error);

	// Request the time-limited AES-128-GCM key for an asset from uro.
	// POSTs `{uuid: p_asset_uuid}` to `URO_PATH_SCRIPT_KEY`, decodes the
	// response via `parse_script_key_json`, and persists the material in
	// OS secure storage via FabricMMOGKeyStore so subsequent calls can
	// short-circuit the network round trip until `KEY_TTL_SECONDS`.
	// Returns OK and populates `r_key` (AES_KEY_BYTES) / `r_iv`
	// (AES_IV_BYTES) on success; on failure sets `r_error`.
	static Error request_asset_key(const String &p_uro_base_url,
			const String &p_asset_uuid,
			PackedByteArray &r_key, PackedByteArray &r_iv,
			String &r_error);

	FabricMMOGAsset() {}
	~FabricMMOGAsset() {}

protected:
	static void _bind_methods();
};
