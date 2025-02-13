/**************************************************************************/
/*  clmul.c                                                               */
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

#include <stdio.h>
#include <x86intrin.h>

int main() {
	__m128i a;
	__m128i b;

	a[1] = 2;
	a[0] = -1284;
	b[1] = 25;
	b[0] = 65535;

	// _mm_clmulepi64_si128 only looks at the least significant bit of each
	__m128i result1 = _mm_clmulepi64_si128(a, b, 0x11);
	__m128i result2 = _mm_clmulepi64_si128(a, b, 0x00);
	__m128i result3 = _mm_clmulepi64_si128(a, b, 0xF2);

	printf("%lld times %lld without a carry bit: %lld\n",
			a[1], b[1], result1[0]);
	printf("%lld times %lld without a carry bit: %lld\n",
			a[0], b[0], result2[0]);
	printf("%lld times %lld without a carry bit: %lld\n",
			a[0], b[1], result3[0]);
	return 0;
}
