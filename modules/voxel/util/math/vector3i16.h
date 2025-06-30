/**************************************************************************/
/*  vector3i16.h                                                          */
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

#include "vector3t.h"
#include <cstdint>
#include <functional>

namespace zylann {

typedef Vector3T<int16_t> Vector3i16;

inline size_t get_hash_st(const zylann::Vector3i16 &v) {
	// TODO Optimization: benchmark this hash, I just wanted one that works
	// static_assert(sizeof(zylann::Vector3i16) <= sizeof(uint64_t));
	const uint64_t m = v.x | (static_cast<uint64_t>(v.y) << 16) | (static_cast<uint64_t>(v.z) << 32);
	return std::hash<uint64_t>{}(m);
}

} // namespace zylann

// For STL
namespace std {
template <>
struct hash<zylann::Vector3i16> {
	size_t operator()(const zylann::Vector3i16 &v) const {
		return zylann::get_hash_st(v);
	}
};
} // namespace std
