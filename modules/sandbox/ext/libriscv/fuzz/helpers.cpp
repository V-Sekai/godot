/**************************************************************************/
/*  helpers.cpp                                                           */
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

#include <climits>

/* It is necessary to link with libgcc when fuzzing.
   See llvm.org/PR30643 for details. */
__attribute__((weak, no_sanitize("undefined"))) extern "C" __int128_t
__muloti4(__int128_t a, __int128_t b, int *overflow) {
	const int N = (int)(sizeof(__int128_t) * CHAR_BIT);
	const __int128_t MIN = (__int128_t)1 << (N - 1);
	const __int128_t MAX = ~MIN;
	*overflow = 0;
	__int128_t result = a * b;
	if (a == MIN) {
		if (b != 0 && b != 1)
			*overflow = 1;
		return result;
	}
	if (b == MIN) {
		if (a != 0 && a != 1)
			*overflow = 1;
		return result;
	}
	__int128_t sa = a >> (N - 1);
	__int128_t abs_a = (a ^ sa) - sa;
	__int128_t sb = b >> (N - 1);
	__int128_t abs_b = (b ^ sb) - sb;
	if (abs_a < 2 || abs_b < 2)
		return result;
	if (sa == sb) {
		if (abs_a > MAX / abs_b)
			*overflow = 1;
	} else {
		if (abs_a > MIN / -abs_b)
			*overflow = 1;
	}
	return result;
}
