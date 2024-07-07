/**************************************************************************/
/*  wasgo_math.h                                                          */
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

#include "wasm_export.h"
#include <math.h>
#include <stdlib.h>

//Math functions for the runtime
//used in the native wrappers
inline double _wasgo_cos(wasm_exec_env_t exec_env, double x) {
	return cos(x);
}
inline double _wasgo_sin(wasm_exec_env_t exec_env, double x) {
	return sin(x);
}
inline double _wasgo_tan(wasm_exec_env_t exec_env, double x) {
	return tan(x);
}
inline double _wasgo_acos(wasm_exec_env_t exec_env, double x) {
	return acos(x);
}
inline double _wasgo_asin(wasm_exec_env_t exec_env, double x) {
	return asin(x);
}
inline double _wasgo_atan(wasm_exec_env_t exec_env, double x) {
	return atan(x);
}
inline double _wasgo_atan2(wasm_exec_env_t exec_env, double x, double y) {
	return atan2(x, y);
}

inline float _wasgo_cosf(wasm_exec_env_t exec_env, float x) {
	return cosf(x);
}
inline float _wasgo_sinf(wasm_exec_env_t exec_env, float x) {
	return sinf(x);
}
inline float _wasgo_tanf(wasm_exec_env_t exec_env, float x) {
	return tanf(x);
}
inline float _wasgo_acosf(wasm_exec_env_t exec_env, float x) {
	return acosf(x);
}
inline float _wasgo_asinf(wasm_exec_env_t exec_env, float x) {
	return asinf(x);
}
inline float _wasgo_atanf(wasm_exec_env_t exec_env, float x) {
	return atanf(x);
}
inline float _wasgo_atan2f(wasm_exec_env_t exec_env, float x, float y) {
	return atan2f(x, y);
}

inline int _wasgo_rand(wasm_exec_env_t exec_env) {
	return rand();
}

// inline float _wasgo_random(wasm_exec_env_t exec_env) {
// 	return random();
// }