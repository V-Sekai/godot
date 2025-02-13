/**************************************************************************/
/*  floating-point.hpp                                                    */
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

#ifndef FLOATING_POINT_HPP
#define FLOATING_POINT_HPP

extern float test_fadd(float a, float b);
extern float test_fsub(float a, float b);
extern float test_fmul(float a, float b);
extern float test_fdiv(float a, float b);
extern float test_fmax(float a, float b);
extern float test_fmin(float a, float b);
extern double test_ftod(float val);
extern float test_dtof(double val);

extern float test_fneg(float val);
extern double test_dneg(double val);

extern float test_fmadd(float a, float b, float c);
extern float test_fmsub(float a, float b, float c);
extern float test_fnmadd(float a, float b, float c);
extern float test_fnmsub(float a, float b, float c);

extern float test_fsqrt(float val);
extern double test_dsqrt(double val);
extern float test_fpow(float val, float);
extern double test_dpow(double val, double);

extern float test_sinf(float val);
extern float test_cosf(float val);
extern float test_tanf(float val);

#endif // FLOATING_POINT_HPP
