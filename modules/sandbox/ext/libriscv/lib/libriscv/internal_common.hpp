/**************************************************************************/
/*  internal_common.hpp                                                   */
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

#ifndef INTERNAL_COMMON_HPP
#define INTERNAL_COMMON_HPP

#ifdef __APPLE__
#include "TargetConditionals.h" // TARGET_* macros
#endif

#ifdef __GNUG__
#define RISCV_NOINLINE __attribute__((noinline))
#define RISCV_UNREACHABLE() __builtin_unreachable()
#define RISCV_EXPORT __attribute__((visibility("default")))
#else
#define RISCV_NOINLINE /* */
#define RISCV_UNREACHABLE() /* */
#ifdef _MSC_VER
#define RISCV_EXPORT __declspec(dllexport)
#else
#define RISCV_EXPORT /* */
#endif
#endif

#ifdef RISCV_32I
#define INSTANTIATE_32_IF_ENABLED(x) template struct x<4>
#else
#define INSTANTIATE_32_IF_ENABLED(x) /* */
#endif

#ifdef RISCV_64I
#define INSTANTIATE_64_IF_ENABLED(x) template struct x<8>
#else
#define INSTANTIATE_64_IF_ENABLED(x) /* */
#endif

#ifdef RISCV_128I
#define INSTANTIATE_128_IF_ENABLED(x) template struct x<16>
#else
#define INSTANTIATE_128_IF_ENABLED(x) /* */
#endif

#ifndef ANTI_FINGERPRINTING_MASK_MICROS
#define ANTI_FINGERPRINTING_MASK_MICROS() ~0x3FFLL
#endif
#ifndef ANTI_FINGERPRINTING_MASK_NANOS
#define ANTI_FINGERPRINTING_MASK_NANOS() ~0xFFFFFLL
#endif

#endif // INTERNAL_COMMON_HPP
