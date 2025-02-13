/**************************************************************************/
/*  floating-point.cpp                                                    */
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

#include "floating-point.hpp"
#include <algorithm>
#include <cmath>

float test_fadd(float a, float b) {
	return a + b;
}
float test_fsub(float a, float b) {
	return a - b;
}
float test_fmul(float a, float b) {
	return a * b;
}
float test_fdiv(float a, float b) {
	return a / b;
}
float test_fmax(float a, float b) {
	return std::max(a, b);
}
float test_fmin(float a, float b) {
	return std::min(a, b);
}
double test_ftod(float val) {
	return (double)val;
}
float test_dtof(double val) {
	return (float)val;
}

float test_fneg(float val) {
	return -val;
}
double test_dneg(double val) {
	return -val;
}

float test_fmadd(float a, float b, float c) {
	return a * b + c;
}
float test_fmsub(float a, float b, float c) {
	return a * b - c;
}
float test_fnmadd(float a, float b, float c) {
	return -(a * b) + c;
}
float test_fnmsub(float a, float b, float c) {
	return -(a * b) - c;
}
float test_dotp(float *a, float *b, unsigned count) {
	float sum = 0.0f;
	for (size_t i = 0; i < count; i++)
		sum += a[i] * b[i];
	return sum;
}

float test_fsqrt(float val) {
	return std::sqrt(val);
}
double test_dsqrt(double val) {
	return std::sqrt(val);
}
float test_fpow(float val, float exp) {
	return std::pow(val, exp);
}
double test_dpow(double val, double exp) {
	return std::pow(val, exp);
}

float test_sinf(float val) {
	return std::sin(val);
}
float test_cosf(float val) {
	return std::cos(val);
}
float test_tanf(float val) {
	return std::tan(val);
}
