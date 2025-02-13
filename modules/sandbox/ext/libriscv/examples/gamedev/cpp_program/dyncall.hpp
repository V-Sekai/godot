/**************************************************************************/
/*  dyncall.hpp                                                           */
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

#ifndef DYNCALL_HPP
#define DYNCALL_HPP
#include <cstddef>

#define DEFINE_DYNCALL(number, name, type)                      \
	asm(".pushsection .text\n"                                  \
		".func sys_" #name "\n"                                 \
		"sys_" #name ":\n"                                      \
		"   .insn i 0b1011011, 0, x0, x0, " #number "\n"        \
		"   ret\n"                                              \
		".endfunc\n"                                            \
		".popsection .text\n");                                 \
	using name##_t = type;                                      \
	extern "C" __attribute__((used, retain)) void sys_##name(); \
	template <typename... Args>                                 \
	static inline auto name(Args &&...args) {                   \
		auto fn = (name##_t *)sys_##name;                       \
		return fn(std::forward<Args>(args)...);                 \
	}

#define EXTERN_DYNCALL(name, type)                              \
	using name##_t = type;                                      \
	extern "C" __attribute__((used, retain)) void sys_##name(); \
	template <typename... Args>                                 \
	static inline auto name(Args &&...args) {                   \
		auto fn = (name##_t *)sys_##name;                       \
		return fn(std::forward<Args>(args)...);                 \
	}

#endif // DYNCALL_HPP
