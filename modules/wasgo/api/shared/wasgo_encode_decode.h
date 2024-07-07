/**************************************************************************/
/*  wasgo_encode_decode.h                                                 */
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

#ifndef WASGO_ENCODE_DECODE_H
#define WASGO_ENCODE_DECODE_H
//Lots of this is copied from Godot's marshalls.h

// #include "reference.h"
// #include "typedefs.h"
#include "wasgoid.h"
#include <limits.h>
#include <cstddef>
#include <string>
// #include "core/variant/variant.h"

/**
 * Miscellaneous helpers for marshaling data types, and encoding
 * in an endian independent way
 */

union MarshallFloat {
	uint32_t i; ///< int
	float f; ///< float
};

union MarshallDouble {
	uint64_t l; ///< long long
	double d; ///< double
};

static inline unsigned int encode_uint16(uint16_t p_uint, WasGoByte *p_arr) {
	for (int i = 0; i < 2; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}

	return sizeof(uint16_t);
}

static inline unsigned int encode_uint32(uint32_t p_uint, WasGoByte *p_arr) {
	for (int i = 0; i < 4; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}

	return sizeof(uint32_t);
}

static inline unsigned int encode_float(float p_float, WasGoByte *p_arr) {
	MarshallFloat mf;
	mf.f = p_float;
	encode_uint32(mf.i, p_arr);

	return sizeof(uint32_t);
}

static inline unsigned int encode_uint64(uint64_t p_uint, WasGoByte *p_arr) {
	for (int i = 0; i < 8; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}

	return sizeof(uint64_t);
}

static inline unsigned int encode_double(double p_double, WasGoByte *p_arr) {
	MarshallDouble md;
	md.d = p_double;
	encode_uint64(md.l, p_arr);

	return sizeof(uint64_t);
}

static inline int encode_cstring(const char *p_string, WasGoByte *p_data) {
	int len = 0;

	while (*p_string) {
		if (p_data) {
			*p_data = (WasGoByte)*p_string;
			p_data++;
		}
		p_string++;
		len++;
	};

	if (p_data)
		*p_data = 0;
	return len + 1;
}

static inline uint16_t decode_uint16(const WasGoByte *p_arr) {
	uint16_t u = 0;

	for (int i = 0; i < 2; i++) {
		uint16_t b = *p_arr;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}

	return u;
}

static inline uint32_t decode_uint32(const WasGoByte *p_arr) {
	uint32_t u = 0;

	for (int i = 0; i < 4; i++) {
		uint32_t b = *p_arr;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}

	return u;
}

static inline float decode_float(const WasGoByte *p_arr) {
	MarshallFloat mf;
	mf.i = decode_uint32(p_arr);
	return mf.f;
}

static inline uint64_t decode_uint64(const WasGoByte *p_arr) {
	uint64_t u = 0;

	for (int i = 0; i < 8; i++) {
		uint64_t b = (*p_arr) & 0xFF;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}

	return u;
}

static inline double decode_double(const WasGoByte *p_arr) {
	MarshallDouble md;
	md.l = decode_uint64(p_arr);
	return md.d;
}

#define _S(a) ((int32_t)a)
#define ERR_FAIL_ADD_OF(a, b) ERR_FAIL_COND(_S(b) < 0 || _S(a) < 0 || _S(a) > INT_MAX - _S(b))
#define ERR_FAIL_MUL_OF(a, b) ERR_FAIL_COND(_S(a) < 0 || _S(b) <= 0 || _S(a) > INT_MAX / _S(b))

#define ENCODE_MASK 0xFF
#define ENCODE_FLAG_64 1 << 16
#define ENCODE_FLAG_OBJECT_AS_ID 1 << 16

// static void _decode_string(WasGoByte *&buf, size_t &len, size_t *r_len, char *&r_string) {
// 	ERR_FAIL_COND(len < 4);

// 	int32_t strlen = decode_uint32(buf);
// 	size_t pad = 0;

// 	// Handle padding
// 	if (strlen % 4) {
// 		pad = 4 - strlen % 4;
// 	}

// 	buf += 4;
// 	len -= 4;

// 	// Ensure buffer is big enough
// 	ERR_FAIL_ADD_OF(strlen, pad);
// 	ERR_FAIL_COND(strlen < 0);
// 	ERR_FAIL_INDEX(strlen + pad, len);

// 	char *str = (char *)buf;
// 	// ERR_FAIL_COND_V(str =(const char *)buf, strlen));
// 	r_string = str;

// 	// Add padding
// 	strlen += pad;

// 	// Update buffer pos, left data count, and return size
// 	buf += strlen;
// 	len -= strlen;
// 	if (r_len) {
// 		(*r_len) += 4 + strlen;
// 	}
// }
// static void _encode_string(const char *utf8, WasGoByte *buf, size_t &r_len) {
// 	size_t length = strlen(utf8);
// 	if (buf) {
// 		encode_uint32(length, buf);
// 		buf += 4;
// 		memcpy(buf, utf8, length);
// 		buf += length;
// 	}

// 	r_len += 4 + length;
// 	while (r_len % 4) {
// 		r_len++; //pad
// 		if (buf) {
// 			*(buf++) = 0;
// 		}
// 	}
// }

// class EncodedObjectAsID : public Reference {
// 	// GDCLASS(EncodedObjectAsID, Reference);

// 	WasGoID id;

// protected:
// 	static void _bind_methods();

// public:
// 	void set_object_id(WasGoID p_id);
// 	WasGoID get_object_id() const;

// 	EncodedObjectAsID();
// };

// Error decode_variant(Variant &r_variant, const WasGoByte *p_buffer, int p_len, int *r_len = NULL, bool p_allow_objects = false);
// Error encode_variant(const Variant &p_variant, WasGoByte *r_buffer, int &r_len, bool p_full_objects = false);

#endif // WASGO_ENCODE_DECODE_H
