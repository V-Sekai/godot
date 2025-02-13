/**************************************************************************/
/*  crc32.hpp                                                             */
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

#ifndef CRC32_HPP
#define CRC32_HPP
#include <array>
#include <cstddef>
#include <cstdint>

namespace riscv {

template <uint32_t POLYNOMIAL>
inline constexpr auto gen_crc32_table() {
	constexpr auto num_iterations = 8;
	auto crc32_table = std::array<uint32_t, 256>{};

	for (auto byte = 0u; byte < crc32_table.size(); ++byte) {
		auto crc = byte;

		for (auto i = 0; i < num_iterations; ++i) {
			auto mask = -(crc & 1);
			crc = (crc >> 1) ^ (POLYNOMIAL & mask);
		}

		crc32_table[byte] = crc;
	}
	return crc32_table;
}

template <uint32_t POLYNOMIAL = 0xEDB88320>
inline constexpr auto crc32(const char *data) {
	constexpr auto crc32_table = gen_crc32_table<POLYNOMIAL>();

	auto crc = 0xFFFFFFFFu;
	for (auto i = 0u; auto c = data[i]; ++i) {
		crc = crc32_table[(crc ^ c) & 0xFF] ^ (crc >> 8);
	}
	return ~crc;
}

template <uint32_t POLYNOMIAL = 0xEDB88320>
inline constexpr auto crc32(uint32_t crc, const void *vdata, const size_t len) {
	constexpr auto crc32_table = gen_crc32_table<POLYNOMIAL>();

	auto *data = (const uint8_t *)vdata;
	for (auto i = 0u; i < len; ++i) {
		crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
	}
	return crc;
}

template <uint32_t POLYNOMIAL = 0xEDB88320>
inline constexpr auto crc32(const void *vdata, const size_t len) {
	return ~crc32<POLYNOMIAL>(0xFFFFFFFF, vdata, len);
}

// CRC32-C with hardware acceleration on amd64
extern uint32_t crc32c(const void *data, size_t);
extern uint32_t crc32c(uint32_t partial, const void *data, size_t);

} //namespace riscv

#endif // CRC32_HPP
