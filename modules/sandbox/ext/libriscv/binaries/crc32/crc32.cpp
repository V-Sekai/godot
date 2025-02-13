/**************************************************************************/
/*  crc32.cpp                                                             */
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

#include <cassert>
#include <cstdint>
#include <cstdio>

static inline int64_t clmul(int64_t x1, int64_t x2) {
	int64_t result;
	asm("clmul %0, %1, %2" : "=r"(result) : "r"(x1), "r"(x2));
	return result;
}
static inline int64_t clmulh(int64_t x1, int64_t x2) {
	int64_t result;
	asm("clmulh %0, %1, %2" : "=r"(result) : "r"(x1), "r"(x2));
	return result;
}
static inline int64_t clmulr(int64_t x1, int64_t x2) {
	int64_t result;
	asm("clmulr %0, %1, %2" : "=r"(result) : "r"(x1), "r"(x2));
	return result;
}

int main() {
	printf("clmul(2, 25) = %ld\n", clmul(2, 25));
	printf("clmul(-1284, 65535) = %ld\n", clmul(-1284, 65535));
	printf("clmul(-1284, 25) = %ld\n", clmul(-1284, 25));

	assert(clmul(2, 25) == 50);
	assert(clmul(-1284, 65535) == 50419284);
	assert(clmul(-1284, 25) == -32036);

	printf("clmulh(2, 25) = %ld\n", clmulh(2, 25));
	printf("clmulh(-1284, 65535) = %ld\n", clmulh(-1284, 65535));
	printf("clmulh(-1284, 25) = %ld\n", clmulh(-1284, 25));

	assert(clmulh(2, 25) == 0);

	printf("clmulr(2, 25) = %ld\n", clmulr(2, 25));
	printf("clmulr(-1284, 65535) = %ld\n", clmulr(-1284, 65535));
	printf("clmulr(-1284, 25) = %ld\n", clmulr(-1284, 25));

	assert(clmulr(2, 25) == 0);

	printf("Tests passed!\n");
}
