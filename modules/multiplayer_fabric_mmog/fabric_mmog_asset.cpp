/**************************************************************************/
/*  fabric_mmog_asset.cpp                                                 */
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

#include "fabric_mmog_asset.h"

#ifdef MODULE_KEYCHAIN_ENABLED
#include "modules/keychain/fabric_mmog_keystore.h"
#endif

#include "core/crypto/crypto_core.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/http_client.h"
#include "core/io/json.h"
#include "core/io/stream_peer_tls.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/variant/variant.h"

#include <zstd.h>

#include <cstring>

void FabricMMOGAsset::_bind_methods() {
	ClassDB::bind_method(D_METHOD("fetch_asset", "store_url", "index_url", "output_dir", "cache_dir"),
			&FabricMMOGAsset::fetch_asset);
	// request_asset_key is C++-only — its out-parameter signature doesn't
	// fit ClassDB binding. When it graduates from stub to real code, expose
	// a Dictionary-returning wrapper instead.

	BIND_CONSTANT(CHUNK_ID_BYTES);
	BIND_CONSTANT(CHUNK_MIN_BYTES);
	BIND_CONSTANT(CHUNK_MAX_BYTES);
	BIND_CONSTANT(REGISTRY_SLOT_BYTES);
	BIND_CONSTANT(REGISTRY_INDEX_ID_BYTES);
	BIND_CONSTANT(REGISTRY_URO_UUID_BYTES);
	BIND_CONSTANT(REGISTRY_ENTRY_BYTES);
	BIND_CONSTANT(AES_KEY_BYTES);
	BIND_CONSTANT(AES_IV_BYTES);
	BIND_CONSTANT(AES_TAG_BYTES);
	BIND_CONSTANT(KEY_TTL_SECONDS);
}

namespace {

// casync/desync binary format constants — see thirdparty/desync/const.go.
constexpr uint64_t CA_FORMAT_INDEX = 0x96824d9c7b129ff9ULL;
constexpr uint64_t CA_FORMAT_TABLE = 0xe75b9e112f17417dULL;
constexpr uint64_t CA_FORMAT_TABLE_TAIL_MARKER = 0x4b4f050e5549ecd1ULL;
constexpr uint64_t CA_FORMAT_SHA512_256 = 0x2000000000000000ULL;
constexpr uint64_t CA_FORMAT_INDEX_SIZE = 48;
constexpr uint64_t CA_MAX_UINT64 = 0xFFFFFFFFFFFFFFFFULL;

// Little-endian cursor with bounds checking.
struct LECursor {
	const uint8_t *data;
	int64_t size;
	int64_t pos = 0;

	bool read_u64(uint64_t &r_value) {
		if (pos + 8 > size) {
			return false;
		}
		r_value = 0;
		for (int i = 0; i < 8; i++) {
			r_value |= uint64_t(data[pos + i]) << (i * 8);
		}
		pos += 8;
		return true;
	}

	bool read_bytes(uint8_t *r_dst, int64_t p_n) {
		if (pos + p_n > size) {
			return false;
		}
		memcpy(r_dst, data + pos, p_n);
		pos += p_n;
		return true;
	}
};

} // namespace

Error FabricMMOGAsset::parse_caibx(const Vector<uint8_t> &p_bytes,
		Vector<CaibxChunk> &r_chunks, String &r_error) {
	r_chunks.clear();
	r_error = String();

	LECursor cur{ p_bytes.ptr(), p_bytes.size(), 0 };

	// ── FormatIndex header ──────────────────────────────────────────────
	uint64_t index_size = 0;
	uint64_t index_type = 0;
	if (!cur.read_u64(index_size) || !cur.read_u64(index_type)) {
		r_error = "caibx too small for FormatIndex header";
		return ERR_PARSE_ERROR;
	}
	if (index_size != CA_FORMAT_INDEX_SIZE) {
		r_error = vformat("FormatIndex size %d, expected 48", (int64_t)index_size);
		return ERR_PARSE_ERROR;
	}
	if (index_type != CA_FORMAT_INDEX) {
		r_error = "not a caibx: FormatIndex magic mismatch";
		return ERR_PARSE_ERROR;
	}

	uint64_t feature_flags = 0;
	uint64_t chunk_size_min = 0;
	uint64_t chunk_size_avg = 0;
	uint64_t chunk_size_max = 0;
	if (!cur.read_u64(feature_flags) || !cur.read_u64(chunk_size_min) ||
			!cur.read_u64(chunk_size_avg) || !cur.read_u64(chunk_size_max)) {
		r_error = "caibx truncated inside FormatIndex body";
		return ERR_PARSE_ERROR;
	}
	if ((feature_flags & CA_FORMAT_SHA512_256) == 0) {
		r_error = "caibx does not use SHA-512/256 chunk IDs";
		return ERR_PARSE_ERROR;
	}

	// ── FormatTable header ──────────────────────────────────────────────
	uint64_t table_size = 0;
	uint64_t table_type = 0;
	if (!cur.read_u64(table_size) || !cur.read_u64(table_type)) {
		r_error = "caibx truncated before FormatTable header";
		return ERR_PARSE_ERROR;
	}
	if (table_size != CA_MAX_UINT64) {
		r_error = "FormatTable size is not MAX_UINT64";
		return ERR_PARSE_ERROR;
	}
	if (table_type != CA_FORMAT_TABLE) {
		r_error = "FormatTable magic mismatch";
		return ERR_PARSE_ERROR;
	}

	// ── Chunk table items ───────────────────────────────────────────────
	uint64_t last_offset = 0;
	for (;;) {
		uint64_t offset = 0;
		if (!cur.read_u64(offset)) {
			r_error = "caibx truncated inside chunk table";
			return ERR_PARSE_ERROR;
		}
		if (offset == 0) {
			break; // Zero offset terminates the item list.
		}
		CaibxChunk chunk;
		if (!cur.read_bytes(chunk.id, CHUNK_ID_BYTES)) {
			r_error = "caibx truncated inside chunk ID";
			return ERR_PARSE_ERROR;
		}
		if (offset < last_offset) {
			r_error = "chunk table offsets not monotonic";
			return ERR_PARSE_ERROR;
		}
		chunk.start = last_offset;
		chunk.size = offset - last_offset;
		if (chunk.size > chunk_size_max) {
			r_error = vformat("chunk size %d exceeds max %d",
					(int64_t)chunk.size, (int64_t)chunk_size_max);
			return ERR_PARSE_ERROR;
		}
		r_chunks.push_back(chunk);
		last_offset = offset;
	}

	// ── Tail ────────────────────────────────────────────────────────────
	uint64_t zero_fill_2 = 0;
	uint64_t index_offset = 0;
	uint64_t tail_table_size = 0;
	uint64_t tail_marker = 0;
	if (!cur.read_u64(zero_fill_2) || !cur.read_u64(index_offset) ||
			!cur.read_u64(tail_table_size) || !cur.read_u64(tail_marker)) {
		r_error = "caibx truncated inside tail marker";
		return ERR_PARSE_ERROR;
	}
	if (zero_fill_2 != 0) {
		r_error = "caibx zero fill after item loop is non-zero";
		return ERR_PARSE_ERROR;
	}
	if (tail_marker != CA_FORMAT_TABLE_TAIL_MARKER) {
		r_error = "caibx tail marker mismatch";
		return ERR_PARSE_ERROR;
	}

	return OK;
}

String FabricMMOGAsset::build_chunk_url(const String &p_store_url,
		const uint8_t p_chunk_id[CHUNK_ID_BYTES]) {
	static const char kHex[] = "0123456789abcdef";
	char hex_buf[CHUNK_ID_BYTES * 2 + 1];
	for (int i = 0; i < CHUNK_ID_BYTES; i++) {
		hex_buf[i * 2] = kHex[(p_chunk_id[i] >> 4) & 0x0F];
		hex_buf[i * 2 + 1] = kHex[p_chunk_id[i] & 0x0F];
	}
	hex_buf[CHUNK_ID_BYTES * 2] = '\0';
	const String hex(hex_buf);

	String base = p_store_url;
	if (base.ends_with("/")) {
		base = base.substr(0, base.length() - 1);
	}
	return base + "/" + hex.substr(0, 4) + "/" + hex + ".cacnk";
}

namespace {

// SHA-512 round constants (FIPS 180-4 §4.2.3).
constexpr uint64_t K512[80] = {
	0x428a2f98d728ae22ULL,
	0x7137449123ef65cdULL,
	0xb5c0fbcfec4d3b2fULL,
	0xe9b5dba58189dbbcULL,
	0x3956c25bf348b538ULL,
	0x59f111f1b605d019ULL,
	0x923f82a4af194f9bULL,
	0xab1c5ed5da6d8118ULL,
	0xd807aa98a3030242ULL,
	0x12835b0145706fbeULL,
	0x243185be4ee4b28cULL,
	0x550c7dc3d5ffb4e2ULL,
	0x72be5d74f27b896fULL,
	0x80deb1fe3b1696b1ULL,
	0x9bdc06a725c71235ULL,
	0xc19bf174cf692694ULL,
	0xe49b69c19ef14ad2ULL,
	0xefbe4786384f25e3ULL,
	0x0fc19dc68b8cd5b5ULL,
	0x240ca1cc77ac9c65ULL,
	0x2de92c6f592b0275ULL,
	0x4a7484aa6ea6e483ULL,
	0x5cb0a9dcbd41fbd4ULL,
	0x76f988da831153b5ULL,
	0x983e5152ee66dfabULL,
	0xa831c66d2db43210ULL,
	0xb00327c898fb213fULL,
	0xbf597fc7beef0ee4ULL,
	0xc6e00bf33da88fc2ULL,
	0xd5a79147930aa725ULL,
	0x06ca6351e003826fULL,
	0x142929670a0e6e70ULL,
	0x27b70a8546d22ffcULL,
	0x2e1b21385c26c926ULL,
	0x4d2c6dfc5ac42aedULL,
	0x53380d139d95b3dfULL,
	0x650a73548baf63deULL,
	0x766a0abb3c77b2a8ULL,
	0x81c2c92e47edaee6ULL,
	0x92722c851482353bULL,
	0xa2bfe8a14cf10364ULL,
	0xa81a664bbc423001ULL,
	0xc24b8b70d0f89791ULL,
	0xc76c51a30654be30ULL,
	0xd192e819d6ef5218ULL,
	0xd69906245565a910ULL,
	0xf40e35855771202aULL,
	0x106aa07032bbd1b8ULL,
	0x19a4c116b8d2d0c8ULL,
	0x1e376c085141ab53ULL,
	0x2748774cdf8eeb99ULL,
	0x34b0bcb5e19b48a8ULL,
	0x391c0cb3c5c95a63ULL,
	0x4ed8aa4ae3418acbULL,
	0x5b9cca4f7763e373ULL,
	0x682e6ff3d6b2b8a3ULL,
	0x748f82ee5defb2fcULL,
	0x78a5636f43172f60ULL,
	0x84c87814a1f0ab72ULL,
	0x8cc702081a6439ecULL,
	0x90befffa23631e28ULL,
	0xa4506cebde82bde9ULL,
	0xbef9a3f7b2c67915ULL,
	0xc67178f2e372532bULL,
	0xca273eceea26619cULL,
	0xd186b8c721c0c207ULL,
	0xeada7dd6cde0eb1eULL,
	0xf57d4f7fee6ed178ULL,
	0x06f067aa72176fbaULL,
	0x0a637dc5a2c898a6ULL,
	0x113f9804bef90daeULL,
	0x1b710b35131c471bULL,
	0x28db77f523047d84ULL,
	0x32caab7b40c72493ULL,
	0x3c9ebe0a15c9bebcULL,
	0x431d67c49c100d4cULL,
	0x4cc5d4becb3e42b6ULL,
	0x597f299cfc657e2aULL,
	0x5fcb6fab3ad6faecULL,
	0x6c44198c4a475817ULL,
};

// SHA-512/256 IV (FIPS 180-4 §5.3.6.2).
constexpr uint64_t IV512_256[8] = {
	0x22312194fc2bf72cULL,
	0x9f555fa3c84c64c2ULL,
	0x2393b86b6f53b151ULL,
	0x963877195940eabdULL,
	0x96283ee2a88effe3ULL,
	0xbe5e1e2553863992ULL,
	0x2b0199fc2c85b8aaULL,
	0x0eb72ddc81c52ca2ULL,
};

static inline uint64_t rotr64(uint64_t p_x, int p_n) {
	return (p_x >> p_n) | (p_x << (64 - p_n));
}

// Process one 128-byte SHA-512 block.
static void sha512_compress(uint64_t s[8], const uint8_t p_block[128]) {
	uint64_t w[80];
	for (int i = 0; i < 16; i++) {
		const uint8_t *b = p_block + i * 8;
		w[i] = (uint64_t(b[0]) << 56) | (uint64_t(b[1]) << 48) |
				(uint64_t(b[2]) << 40) | (uint64_t(b[3]) << 32) |
				(uint64_t(b[4]) << 24) | (uint64_t(b[5]) << 16) |
				(uint64_t(b[6]) << 8) | uint64_t(b[7]);
	}
	for (int i = 16; i < 80; i++) {
		const uint64_t s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7);
		const uint64_t s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}

	uint64_t a = s[0], b = s[1], c = s[2], d = s[3];
	uint64_t e = s[4], f = s[5], g = s[6], h = s[7];
	for (int i = 0; i < 80; i++) {
		const uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
		const uint64_t ch = (e & f) ^ (~e & g);
		const uint64_t temp1 = h + S1 + ch + K512[i] + w[i];
		const uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
		const uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
		const uint64_t temp2 = S0 + maj;
		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}
	s[0] += a;
	s[1] += b;
	s[2] += c;
	s[3] += d;
	s[4] += e;
	s[5] += f;
	s[6] += g;
	s[7] += h;
}

} // namespace

void FabricMMOGAsset::sha512_256(const uint8_t *p_data, int64_t p_len,
		uint8_t r_digest[CHUNK_ID_BYTES]) {
	uint64_t s[8];
	memcpy(s, IV512_256, sizeof(IV512_256));

	// Process all whole 128-byte blocks.
	int64_t remaining = p_len;
	const uint8_t *cur = p_data;
	while (remaining >= 128) {
		sha512_compress(s, cur);
		cur += 128;
		remaining -= 128;
	}

	// Final block(s): copy the tail, append 0x80, zero-pad, then the
	// 128-bit big-endian bit length in the last 16 bytes. If the tail +
	// 0x80 + length doesn't fit, one extra block is processed.
	uint8_t tail[256];
	memcpy(tail, cur, remaining);
	tail[remaining] = 0x80;
	int64_t tail_len = remaining + 1;
	int64_t block_count = 1;
	if (tail_len > 112) {
		block_count = 2;
	}
	const int64_t pad_to = block_count * 128;
	memset(tail + tail_len, 0, pad_to - tail_len);

	// 128-bit bit-length, big-endian. p_len fits in 64 bits, so the top
	// 64 bits of the length field are always zero.
	const uint64_t bit_len = uint64_t(p_len) * 8ULL;
	uint8_t *len_field = tail + pad_to - 8;
	for (int i = 0; i < 8; i++) {
		len_field[7 - i] = uint8_t((bit_len >> (i * 8)) & 0xFF);
	}

	for (int64_t i = 0; i < block_count; i++) {
		sha512_compress(s, tail + i * 128);
	}

	// Output = first 256 bits of the state, big-endian.
	for (int i = 0; i < 4; i++) {
		const uint64_t word = s[i];
		for (int j = 0; j < 8; j++) {
			r_digest[i * 8 + j] = uint8_t((word >> ((7 - j) * 8)) & 0xFF);
		}
	}
}

Error FabricMMOGAsset::decompress_and_verify_chunk(
		const Vector<uint8_t> &p_compressed,
		const uint8_t p_expected_id[CHUNK_ID_BYTES],
		Vector<uint8_t> &r_decompressed, String &r_error) {
	r_decompressed.clear();
	r_error = String();

	if (p_compressed.is_empty()) {
		r_error = "compressed chunk is empty";
		return ERR_PARSE_ERROR;
	}

	// Hard upper bound so a hostile frame can't force a multi-gigabyte
	// allocation. 16× CHUNK_MAX_BYTES is well above anything desync would
	// legitimately produce.
	const uint64_t max_decompressed = uint64_t(CHUNK_MAX_BYTES) * 16ULL;

	// Probe the frame header for the decompressed size. The Go `desync`
	// tool writes zstd frames with the content-size flag CLEARED, so we
	// have to handle ZSTD_CONTENTSIZE_UNKNOWN too — fall back to streaming
	// decompression into a capped buffer.
	const unsigned long long content_size = ZSTD_getFrameContentSize(
			p_compressed.ptr(), p_compressed.size());
	if (content_size == ZSTD_CONTENTSIZE_ERROR) {
		r_error = "zstd frame header invalid";
		return ERR_PARSE_ERROR;
	}

	size_t decompressed_size = 0;
	if (content_size != ZSTD_CONTENTSIZE_UNKNOWN) {
		if (content_size > max_decompressed) {
			r_error = vformat("zstd frame content size %d exceeds cap %d",
					int64_t(content_size), int64_t(max_decompressed));
			return ERR_PARSE_ERROR;
		}

		r_decompressed.resize(int(content_size));
		decompressed_size = ZSTD_decompress(
				r_decompressed.ptrw(), r_decompressed.size(),
				p_compressed.ptr(), p_compressed.size());
		if (ZSTD_isError(decompressed_size)) {
			r_decompressed.clear();
			r_error = vformat("zstd decompress failed: %s",
					ZSTD_getErrorName(decompressed_size));
			return ERR_PARSE_ERROR;
		}
		if (int64_t(decompressed_size) != int64_t(content_size)) {
			r_decompressed.clear();
			r_error = "zstd decompressed size differs from frame header";
			return ERR_PARSE_ERROR;
		}
	} else {
		// Streaming path for content-size-unknown frames. Grow an output
		// buffer up to `max_decompressed`, aborting if a hostile frame
		// tries to blow past the cap.
		ZSTD_DCtx *dctx = ZSTD_createDCtx();
		if (dctx == nullptr) {
			r_error = "could not create zstd decompression context";
			return ERR_OUT_OF_MEMORY;
		}

		const size_t out_block_size = ZSTD_DStreamOutSize();
		r_decompressed.resize(int(out_block_size));
		size_t total_out = 0;

		ZSTD_inBuffer in_buf{ p_compressed.ptr(), size_t(p_compressed.size()), 0 };
		size_t ret = 0;
		while (in_buf.pos < in_buf.size) {
			if (total_out + out_block_size > max_decompressed) {
				ZSTD_freeDCtx(dctx);
				r_decompressed.clear();
				r_error = vformat(
						"zstd streaming output exceeds cap %d",
						int64_t(max_decompressed));
				return ERR_PARSE_ERROR;
			}
			if (int64_t(total_out + out_block_size) > int64_t(r_decompressed.size())) {
				r_decompressed.resize(int(total_out + out_block_size));
			}

			ZSTD_outBuffer out_buf{
				r_decompressed.ptrw() + total_out, out_block_size, 0
			};
			ret = ZSTD_decompressStream(dctx, &out_buf, &in_buf);
			if (ZSTD_isError(ret)) {
				ZSTD_freeDCtx(dctx);
				r_decompressed.clear();
				r_error = vformat("zstd stream decompress failed: %s",
						ZSTD_getErrorName(ret));
				return ERR_PARSE_ERROR;
			}
			total_out += out_buf.pos;

			if (ret == 0) {
				// Frame complete.
				break;
			}
		}
		ZSTD_freeDCtx(dctx);

		if (ret != 0) {
			r_decompressed.clear();
			r_error = "zstd stream ended mid-frame";
			return ERR_PARSE_ERROR;
		}
		r_decompressed.resize(int(total_out));
		decompressed_size = total_out;
	}
	(void)decompressed_size;

	uint8_t actual_id[CHUNK_ID_BYTES];
	sha512_256(r_decompressed.ptr(), r_decompressed.size(), actual_id);
	if (memcmp(actual_id, p_expected_id, CHUNK_ID_BYTES) != 0) {
		r_decompressed.clear();
		r_error = "chunk SHA-512/256 does not match expected ID";
		return ERR_INVALID_DATA;
	}

	return OK;
}

String FabricMMOGAsset::hex_from_id(const uint8_t p_chunk_id[CHUNK_ID_BYTES]) {
	static const char kHex[] = "0123456789abcdef";
	String out;
	for (int i = 0; i < CHUNK_ID_BYTES; i++) {
		char pair[3] = {
			kHex[(p_chunk_id[i] >> 4) & 0x0F],
			kHex[p_chunk_id[i] & 0x0F],
			0,
		};
		out += pair;
	}
	return out;
}

Error FabricMMOGAsset::assemble_from_caibx(
		const Vector<uint8_t> &p_caibx_bytes,
		const HashMap<String, Vector<uint8_t>> &p_chunks_by_hex,
		Vector<uint8_t> &r_output, String &r_error) {
	r_output.clear();
	r_error = String();

	Vector<CaibxChunk> chunks;
	const Error parse_err = parse_caibx(p_caibx_bytes, chunks, r_error);
	if (parse_err != OK) {
		return parse_err;
	}

	uint64_t total = 0;
	for (int i = 0; i < chunks.size(); i++) {
		total += chunks[i].size;
	}
	if (total > uint64_t(INT32_MAX)) {
		r_error = "assembled asset exceeds 2 GB limit";
		return ERR_OUT_OF_MEMORY;
	}
	r_output.resize(int(total));

	for (int i = 0; i < chunks.size(); i++) {
		const CaibxChunk &chunk = chunks[i];
		const String hex = hex_from_id(chunk.id);

		const HashMap<String, Vector<uint8_t>>::ConstIterator it =
				p_chunks_by_hex.find(hex);
		if (it == p_chunks_by_hex.end()) {
			r_output.clear();
			r_error = vformat("chunk %s missing from fetch map", hex);
			return ERR_FILE_NOT_FOUND;
		}

		Vector<uint8_t> decompressed;
		const Error verify_err = decompress_and_verify_chunk(
				it->value, chunk.id, decompressed, r_error);
		if (verify_err != OK) {
			r_output.clear();
			return verify_err;
		}

		if (uint64_t(decompressed.size()) != chunk.size) {
			r_output.clear();
			r_error = vformat(
					"chunk %s decompressed size %d disagrees with index size %d",
					hex, decompressed.size(), int(chunk.size));
			return ERR_INVALID_DATA;
		}

		memcpy(r_output.ptrw() + chunk.start, decompressed.ptr(),
				decompressed.size());
	}

	return OK;
}

Error FabricMMOGAsset::http_get_blocking(const String &p_url,
		Vector<uint8_t> &r_body, String &r_error) {
	return http_request_blocking("GET", p_url, Vector<uint8_t>(),
			String(), r_body, r_error);
}

Error FabricMMOGAsset::http_request_blocking(const String &p_method,
		const String &p_url,
		const Vector<uint8_t> &p_body,
		const String &p_content_type,
		Vector<uint8_t> &r_body, String &r_error) {
	r_body.clear();
	r_error = String();

	HTTPClient::Method method;
	if (p_method == "GET") {
		method = HTTPClient::METHOD_GET;
	} else if (p_method == "POST") {
		method = HTTPClient::METHOD_POST;
	} else {
		r_error = vformat("Unsupported HTTP method: %s", p_method);
		return ERR_INVALID_PARAMETER;
	}

	String url = p_url;
	for (int redirect = 0; redirect < 5; redirect++) {
		String scheme;
		String host;
		int port = 0;
		String request_string;
		String fragment;
		const Error parse_err = url.parse_url(
				scheme, host, port, request_string, fragment);
		if (parse_err != OK) {
			r_error = vformat("Could not parse URL: %s", url);
			return parse_err;
		}

		bool use_tls = false;
		if (scheme == "https://") {
			use_tls = true;
		} else if (scheme != "http://") {
			r_error = vformat("Unsupported URL scheme: %s", scheme);
			return ERR_INVALID_PARAMETER;
		}
		if (port == 0) {
			port = use_tls ? 443 : 80;
		}
		if (request_string.is_empty()) {
			request_string = "/";
		}

		Ref<HTTPClient> client = HTTPClient::create();
		if (client.is_null()) {
			r_error = "Could not create HTTPClient";
			return ERR_UNAVAILABLE;
		}

		Ref<TLSOptions> tls_opts;
		if (use_tls) {
			tls_opts = TLSOptions::client();
		}
		Error err = client->connect_to_host(host, port, tls_opts);
		if (err != OK) {
			r_error = vformat("connect_to_host failed: %d", int(err));
			return err;
		}

		while (true) {
			const HTTPClient::Status status = client->get_status();
			if (status == HTTPClient::STATUS_CONNECTED) {
				break;
			}
			if (status == HTTPClient::STATUS_CANT_RESOLVE ||
					status == HTTPClient::STATUS_CANT_CONNECT ||
					status == HTTPClient::STATUS_CONNECTION_ERROR ||
					status == HTTPClient::STATUS_TLS_HANDSHAKE_ERROR) {
				r_error = vformat("HTTP connect failed with status %d", int(status));
				return ERR_CANT_CONNECT;
			}
			client->poll();
			OS::get_singleton()->delay_usec(1000);
		}

		Vector<String> headers;
		headers.push_back(vformat("Host: %s", host));
		headers.push_back("Accept: */*");
		if (method == HTTPClient::METHOD_POST) {
			if (!p_content_type.is_empty()) {
				headers.push_back(vformat("Content-Type: %s", p_content_type));
			}
			headers.push_back(vformat("Content-Length: %d", p_body.size()));
		}
		err = client->request(method, request_string, headers,
				p_body.is_empty() ? nullptr : p_body.ptr(), p_body.size());
		if (err != OK) {
			r_error = vformat("HTTP request failed: %d", int(err));
			return err;
		}

		while (true) {
			const HTTPClient::Status status = client->get_status();
			if (status == HTTPClient::STATUS_BODY ||
					status == HTTPClient::STATUS_CONNECTED) {
				break;
			}
			if (status == HTTPClient::STATUS_DISCONNECTED ||
					status == HTTPClient::STATUS_CONNECTION_ERROR ||
					status == HTTPClient::STATUS_TLS_HANDSHAKE_ERROR) {
				r_error = vformat("HTTP request aborted with status %d", int(status));
				return ERR_CANT_CONNECT;
			}
			client->poll();
			OS::get_singleton()->delay_usec(1000);
		}

		const int response_code = client->get_response_code();

		if (response_code >= 300 && response_code < 400) {
			List<String> response_headers;
			client->get_response_headers(&response_headers);
			String location;
			for (const String &header : response_headers) {
				if (header.to_lower().begins_with("location:")) {
					location = header.substr(9).strip_edges();
					break;
				}
			}
			client->close();
			if (location.is_empty()) {
				r_error = vformat("HTTP %d with no Location header", response_code);
				return ERR_INVALID_DATA;
			}
			if (!location.begins_with("http://") && !location.begins_with("https://")) {
				location = vformat("%s%s%s", scheme, host, location);
			}
			url = location;
			continue;
		}

		if (response_code < 200 || response_code >= 300) {
			client->close();
			r_error = vformat("HTTP %d for %s", response_code, url);
			return ERR_FILE_CANT_OPEN;
		}

		while (client->get_status() == HTTPClient::STATUS_BODY) {
			client->poll();
			const PackedByteArray chunk = client->read_response_body_chunk();
			if (chunk.size() > 0) {
				const int before = r_body.size();
				r_body.resize(before + chunk.size());
				memcpy(r_body.ptrw() + before, chunk.ptr(), chunk.size());
			} else {
				OS::get_singleton()->delay_usec(1000);
			}
		}
		client->close();
		return OK;
	}

	r_error = "Too many HTTP redirects";
	return ERR_CANT_RESOLVE;
}

String FabricMMOGAsset::fetch_asset(const String &p_store_url,
		const String &p_index_url,
		const String &p_output_dir,
		const String &p_cache_dir) {
	String error;

	// 1. GET the caibx/caidx index file.
	Vector<uint8_t> caibx_bytes;
	if (http_get_blocking(p_index_url, caibx_bytes, error) != OK) {
		ERR_PRINT(vformat("fetch_asset: index GET failed: %s", error));
		return String();
	}

	// 2. Parse it to learn the chunk list.
	Vector<CaibxChunk> chunks;
	if (parse_caibx(caibx_bytes, chunks, error) != OK) {
		ERR_PRINT(vformat("fetch_asset: parse_caibx failed: %s", error));
		return String();
	}

	// 3. Fetch every unique chunk (dedup by hex ID), honoring any cached
	//    copies under p_cache_dir and writing new ones back.
	if (!p_cache_dir.is_empty() && !DirAccess::dir_exists_absolute(p_cache_dir)) {
		DirAccess::make_dir_recursive_absolute(p_cache_dir);
	}

	HashMap<String, Vector<uint8_t>> chunks_by_hex;
	for (int i = 0; i < chunks.size(); i++) {
		const String hex = hex_from_id(chunks[i].id);
		if (chunks_by_hex.has(hex)) {
			continue;
		}

		const String cache_path = p_cache_dir.is_empty()
				? String()
				: p_cache_dir.path_join(hex + ".cacnk");

		Vector<uint8_t> compressed;
		if (!cache_path.is_empty() && FileAccess::exists(cache_path)) {
			Ref<FileAccess> cached = FileAccess::open(cache_path, FileAccess::READ);
			if (cached.is_valid()) {
				const int64_t len = cached->get_length();
				compressed.resize(int(len));
				cached->get_buffer(compressed.ptrw(), len);
			}
		}

		if (compressed.is_empty()) {
			const String chunk_url = build_chunk_url(p_store_url, chunks[i].id);
			if (http_get_blocking(chunk_url, compressed, error) != OK) {
				ERR_PRINT(vformat("fetch_asset: chunk %s GET failed: %s", hex, error));
				return String();
			}
			if (!cache_path.is_empty()) {
				Ref<FileAccess> out = FileAccess::open(cache_path, FileAccess::WRITE);
				if (out.is_valid()) {
					out->store_buffer(compressed.ptr(), compressed.size());
				}
			}
		}

		chunks_by_hex[hex] = compressed;
	}

	// 4. Reassemble via the pure driver.
	Vector<uint8_t> output;
	if (assemble_from_caibx(caibx_bytes, chunks_by_hex, output, error) != OK) {
		ERR_PRINT(vformat("fetch_asset: assemble failed: %s", error));
		return String();
	}

	// 5. Write the reassembled bytes under p_output_dir, using the index
	//    filename with its `.caibx` / `.caidx` suffix stripped.
	String basename = p_index_url.get_file();
	const String ext = basename.get_extension().to_lower();
	if (ext == "caibx" || ext == "caidx") {
		basename = basename.get_basename();
	}
	if (basename.is_empty()) {
		basename = "asset.bin";
	}

	if (!p_output_dir.is_empty() && !DirAccess::dir_exists_absolute(p_output_dir)) {
		DirAccess::make_dir_recursive_absolute(p_output_dir);
	}
	const String output_path = p_output_dir.is_empty()
			? basename
			: p_output_dir.path_join(basename);

	Ref<FileAccess> out_file = FileAccess::open(output_path, FileAccess::WRITE);
	if (out_file.is_null()) {
		ERR_PRINT(vformat("fetch_asset: could not open output %s", output_path));
		return String();
	}
	out_file->store_buffer(output.ptr(), output.size());
	out_file->close();

	return output_path;
}

Error FabricMMOGAsset::parse_manifest_json(const String &p_json,
		Vector<CaibxChunk> &r_chunks, String &r_error) {
	r_chunks.clear();
	r_error = String();

	Variant parsed;
	{
		JSON json;
		const Error parse_err = json.parse(p_json);
		if (parse_err != OK) {
			r_error = vformat("manifest JSON parse error at line %d: %s",
					json.get_error_line(), json.get_error_message());
			return ERR_INVALID_DATA;
		}
		parsed = json.get_data();
	}

	if (parsed.get_type() != Variant::DICTIONARY) {
		r_error = "manifest root is not an object";
		return ERR_INVALID_DATA;
	}
	const Dictionary root = parsed;
	if (!root.has("chunks")) {
		r_error = "manifest missing 'chunks' key";
		return ERR_INVALID_DATA;
	}
	const Variant chunks_var = root["chunks"];
	if (chunks_var.get_type() != Variant::ARRAY) {
		r_error = "manifest 'chunks' is not an array";
		return ERR_INVALID_DATA;
	}
	const Array chunks_arr = chunks_var;

	for (int i = 0; i < chunks_arr.size(); i++) {
		const Variant entry_var = chunks_arr[i];
		if (entry_var.get_type() != Variant::DICTIONARY) {
			r_error = vformat("manifest chunk %d is not an object", i);
			r_chunks.clear();
			return ERR_INVALID_DATA;
		}
		const Dictionary entry = entry_var;
		if (!entry.has("id") || !entry.has("start") || !entry.has("size")) {
			r_error = vformat("manifest chunk %d missing id/start/size", i);
			r_chunks.clear();
			return ERR_INVALID_DATA;
		}
		const String id_hex = entry["id"];
		if (id_hex.length() != CHUNK_ID_BYTES * 2) {
			r_error = vformat("manifest chunk %d id length %d != %d",
					i, id_hex.length(), CHUNK_ID_BYTES * 2);
			r_chunks.clear();
			return ERR_INVALID_DATA;
		}

		CaibxChunk chunk;
		for (int b = 0; b < CHUNK_ID_BYTES; b++) {
			const char32_t hi = id_hex[b * 2];
			const char32_t lo = id_hex[b * 2 + 1];
			const int hi_v = (hi >= '0' && hi <= '9') ? (hi - '0')
					: (hi >= 'a' && hi <= 'f')		  ? (hi - 'a' + 10)
					: (hi >= 'A' && hi <= 'F')		  ? (hi - 'A' + 10)
													  : -1;
			const int lo_v = (lo >= '0' && lo <= '9') ? (lo - '0')
					: (lo >= 'a' && lo <= 'f')		  ? (lo - 'a' + 10)
					: (lo >= 'A' && lo <= 'F')		  ? (lo - 'A' + 10)
													  : -1;
			if (hi_v < 0 || lo_v < 0) {
				r_error = vformat("manifest chunk %d id is not hex", i);
				r_chunks.clear();
				return ERR_INVALID_DATA;
			}
			chunk.id[b] = uint8_t((hi_v << 4) | lo_v);
		}
		chunk.start = uint64_t(int64_t(entry["start"]));
		chunk.size = uint64_t(int64_t(entry["size"]));
		r_chunks.push_back(chunk);
	}

	return OK;
}

Error FabricMMOGAsset::acl_check(const String &p_uro_base_url,
		const String &p_object,
		const String &p_relation,
		const String &p_subject,
		bool &r_allowed, String &r_error) {
	r_allowed = false;
	r_error = String();

	String base = p_uro_base_url;
	while (base.ends_with("/")) {
		base = base.substr(0, base.length() - 1);
	}
	const String url = base + "/acl/check";

	Dictionary request_body;
	request_body["object"] = p_object;
	request_body["relation"] = p_relation;
	request_body["subject"] = p_subject;
	const String body_text = JSON::stringify(request_body);
	const CharString body_utf8 = body_text.utf8();
	Vector<uint8_t> body;
	body.resize(body_utf8.length());
	memcpy(body.ptrw(), body_utf8.get_data(), body_utf8.length());

	Vector<uint8_t> response;
	const Error http_err = http_request_blocking("POST", url, body,
			"application/json", response, r_error);
	if (http_err != OK) {
		return http_err;
	}

	const String response_text = String::utf8(
			reinterpret_cast<const char *>(response.ptr()), response.size());
	JSON json;
	const Error parse_err = json.parse(response_text);
	if (parse_err != OK) {
		r_error = vformat("acl_check parse error at line %d: %s",
				json.get_error_line(), json.get_error_message());
		return ERR_INVALID_DATA;
	}
	const Variant parsed = json.get_data();
	if (parsed.get_type() != Variant::DICTIONARY) {
		r_error = "acl_check response is not an object";
		return ERR_INVALID_DATA;
	}
	const Dictionary root = parsed;
	if (!root.has("allowed")) {
		r_error = "acl_check response missing 'allowed' key";
		return ERR_INVALID_DATA;
	}
	r_allowed = bool(root["allowed"]);
	return OK;
}

Error FabricMMOGAsset::request_manifest(const String &p_uro_base_url,
		const String &p_asset_id,
		Vector<CaibxChunk> &r_chunks, String &r_error) {
	r_chunks.clear();
	r_error = String();

	String base = p_uro_base_url;
	while (base.ends_with("/")) {
		base = base.substr(0, base.length() - 1);
	}
	const String url = vformat("%s/storage/%s/manifest", base,
			p_asset_id.uri_encode());

	const CharString body_utf8 = String("{}").utf8();
	Vector<uint8_t> body;
	body.resize(body_utf8.length());
	memcpy(body.ptrw(), body_utf8.get_data(), body_utf8.length());

	Vector<uint8_t> response;
	const Error http_err = http_request_blocking("POST", url, body,
			"application/json", response, r_error);
	if (http_err != OK) {
		return http_err;
	}

	const String response_text = String::utf8(
			reinterpret_cast<const char *>(response.ptr()), response.size());
	return parse_manifest_json(response_text, r_chunks, r_error);
}

static Error _decode_base64_field(const String &p_b64, int p_expected_len,
		const char *p_field_name, PackedByteArray &r_out, String &r_error) {
	const CharString b64_utf8 = p_b64.utf8();
	r_out.resize(p_expected_len);
	size_t decoded_len = 0;
	const Error err = CryptoCore::b64_decode(r_out.ptrw(), r_out.size(),
			&decoded_len,
			reinterpret_cast<const uint8_t *>(b64_utf8.get_data()),
			b64_utf8.length());
	if (err != OK) {
		r_error = vformat("script key '%s' is not valid base64", p_field_name);
		r_out.resize(0);
		return ERR_INVALID_DATA;
	}
	if (int(decoded_len) != p_expected_len) {
		r_error = vformat("script key '%s' length %d != %d",
				p_field_name, int(decoded_len), p_expected_len);
		r_out.resize(0);
		return ERR_INVALID_DATA;
	}
	return OK;
}

Error FabricMMOGAsset::parse_script_key_json(const String &p_json,
		PackedByteArray &r_key, PackedByteArray &r_iv,
		uint64_t &r_ttl, String &r_error) {
	r_key.resize(0);
	r_iv.resize(0);
	r_ttl = 0;
	r_error = String();

	JSON json;
	const Error parse_err = json.parse(p_json);
	if (parse_err != OK) {
		r_error = vformat("script key JSON parse error at line %d: %s",
				json.get_error_line(), json.get_error_message());
		return ERR_INVALID_DATA;
	}
	const Variant parsed = json.get_data();
	if (parsed.get_type() != Variant::DICTIONARY) {
		r_error = "script key root is not an object";
		return ERR_INVALID_DATA;
	}
	const Dictionary root = parsed;
	if (!root.has("key") || !root.has("iv") || !root.has("ttl")) {
		r_error = "script key response missing key/iv/ttl";
		return ERR_INVALID_DATA;
	}

	const String key_b64 = root["key"];
	if (_decode_base64_field(key_b64, AES_KEY_BYTES, "key", r_key, r_error) != OK) {
		return ERR_INVALID_DATA;
	}
	const String iv_b64 = root["iv"];
	if (_decode_base64_field(iv_b64, AES_IV_BYTES, "iv", r_iv, r_error) != OK) {
		r_key.resize(0);
		return ERR_INVALID_DATA;
	}
	r_ttl = uint64_t(int64_t(root["ttl"]));
	return OK;
}

Error FabricMMOGAsset::request_asset_key(const String &p_uro_base_url,
		const String &p_asset_uuid,
		PackedByteArray &r_key, PackedByteArray &r_iv,
		String &r_error) {
	r_key.resize(0);
	r_iv.resize(0);
	r_error = String();

#ifdef MODULE_KEYCHAIN_ENABLED
	// Cache short-circuit: a live, non-expired entry in the OS key store
	// lets us skip the network round trip entirely.
	{
		String cache_error;
		const Error cache_err = FabricMMOGKeyStore::get(p_asset_uuid,
				r_key, r_iv, cache_error);
		if (cache_err == OK) {
			return OK;
		}
		r_key.resize(0);
		r_iv.resize(0);
	}
#endif

	String base = p_uro_base_url;
	while (base.ends_with("/")) {
		base = base.substr(0, base.length() - 1);
	}
	const String url = base + String(URO_PATH_SCRIPT_KEY);

	Dictionary request_body;
	request_body["uuid"] = p_asset_uuid;
	const String body_text = JSON::stringify(request_body);
	const CharString body_utf8 = body_text.utf8();
	Vector<uint8_t> body;
	body.resize(body_utf8.length());
	memcpy(body.ptrw(), body_utf8.get_data(), body_utf8.length());

	Vector<uint8_t> response;
	const Error http_err = http_request_blocking("POST", url, body,
			"application/json", response, r_error);
	if (http_err != OK) {
		return http_err;
	}

	const String response_text = String::utf8(
			reinterpret_cast<const char *>(response.ptr()), response.size());
	uint64_t ttl = 0;
	const Error parse_err = parse_script_key_json(response_text, r_key, r_iv,
			ttl, r_error);
	if (parse_err != OK) {
		return parse_err;
	}

#ifdef MODULE_KEYCHAIN_ENABLED
	// Best-effort persist: failure to write the OS key store shouldn't
	// fail the request — the caller already has the material in hand.
	String cache_error;
	FabricMMOGKeyStore::put(p_asset_uuid, r_key, r_iv, cache_error);
#endif
	return OK;
}
