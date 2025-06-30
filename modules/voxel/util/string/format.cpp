/**************************************************************************/
/*  format.cpp                                                            */
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

#include "format.h"

namespace zylann {

#ifdef DEV_ENABLED

StdString to_hex_table(Span<const uint8_t> data) {
	StdStringStream ss;
	struct L {
		static inline char to_hex(uint8_t nibble) {
			if (nibble < 10) {
				return '0' + nibble;
			} else {
				return 'a' + (nibble - 10);
			}
		}
	};
	ss << "---";
	ss << std::endl;
	ss << "Data size: ";
	ss << data.size();
	for (unsigned int i = 0; i < data.size(); ++i) {
		if ((i % 16) == 0) {
			ss << std::endl;
			ss << i;
			const unsigned int margin = 6;
			unsigned int p = 10;
			for (unsigned int t = 0; t < margin; ++t) {
				if (i < p) {
					ss << ' ';
				}
				p *= 10;
			}
			ss << " | ";
		}
		const uint8_t b = data[i];
		const uint8_t low_nibble = b & 0xf;
		const uint8_t high_nibble = (b >> 4) & 0xf;
		ss << L::to_hex(high_nibble);
		ss << L::to_hex(low_nibble);
		ss << " ";
	}
	ss << std::endl;
	ss << "---";
	return ss.str();
}

#endif

} // namespace zylann
