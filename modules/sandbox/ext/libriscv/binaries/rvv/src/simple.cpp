/**************************************************************************/
/*  simple.cpp                                                            */
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

#include <array>
using size_t = std::size_t;

union alignas(32) v256 {
	float f[8];

	static inline void zero_v1() {
		asm volatile("vmv.v.i v1, 0");
	}
	static inline void splat_v1(float f) {
		asm volatile("vfmv.s.f     v1, %0" ::"f"(f));
	}
	static inline float sum_v1() {
		float fa0;
		asm volatile("vfredusum.vs v1, v0, v1");
		asm volatile("vfmv.f.s     %0, v1" : "=f"(fa0));
		return fa0;
	}
};

using dot_array = std::array<float, 4096>;
static constexpr dot_array initialize_array(float f) {
	dot_array array{};
	for (auto &val : array)
		val = f;
	return array;
}

int main() {
//#define CONSTINIT
#ifdef CONSTINIT
	alignas(32) static constinit dot_array floats_a = initialize_array(2.0f);
	alignas(32) static constinit dot_array floats_b = initialize_array(2.0f);
#else
	alignas(32) dot_array floats_a;
	alignas(32) dot_array floats_b;
	// Store 2.0f in all elements in both arrays
	v256::splat_v1(2.0f);
	for (size_t i = 0; i < floats_a.size(); i += 16) {
		asm("vse32.v v1, %0" : "=m"(floats_a[i + 0]) : "r"(&floats_a[i + 0]));
		asm("vse32.v v1, %0" : "=m"(floats_a[i + 8]) : "r"(&floats_a[i + 8]));
		asm("vse32.v v1, %0" : "=m"(floats_b[i + 0]) : "r"(&floats_b[i + 0]));
		asm("vse32.v v1, %0" : "=m"(floats_b[i + 8]) : "r"(&floats_b[i + 8]));
	}
#endif

	// Perform RVV dot-product
	v256::zero_v1();
	for (size_t i = 0; i < floats_a.size(); i += 16) {
		v256 *a = (v256 *)&floats_a[i];
		v256 *b = (v256 *)&floats_b[i];
		v256 *c = (v256 *)&floats_a[i + 8];
		v256 *d = (v256 *)&floats_b[i + 8];

		asm("vle32.v v2, %1"
				:
				: "r"(a->f), "m"(a->f[0]));
		asm("vle32.v v3, %1"
				:
				: "r"(b->f), "m"(b->f[0]));

		asm("vfmacc.vv v1, v2, v3");

		asm("vle32.v v2, %1"
				:
				: "r"(c->f), "m"(c->f[0]));
		asm("vle32.v v3, %1"
				:
				: "r"(d->f), "m"(d->f[0]));

		asm("vfmacc.vv v1, v2, v3");
	}
	// Sum elements and return (as integer)
	return v256::sum_v1();
}
