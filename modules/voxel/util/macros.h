/**************************************************************************/
/*  macros.h                                                              */
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

#pragma once

// Macros I couldn't put anywhere specific

// Tell the compiler to favor a certain branch of a condition.
// Until C++20 can be used with the [[likely]] and [[unlikely]] attributes.
#if defined(__GNUC__)
#define ZN_LIKELY(x) __builtin_expect(!!(x), 1)
#define ZN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ZN_LIKELY(x) x
#define ZN_UNLIKELY(x) x
#endif

#define ZN_INTERNAL_CONCAT(x, y) x##y
// Helper to concatenate macro arguments if one of them is itself a macro like `__LINE__`,
// otherwise doing `x##y` directly would not expand the arguments that are a macro
// https://stackoverflow.com/questions/1597007/creating-c-macro-with-and-line-token-concatenation-with-positioning-macr
#define ZN_CONCAT(x, y) ZN_INTERNAL_CONCAT(x, y)

// Gets the name of a class as a C-string with static lifetime, and causes a compiling error if the class doesn't exist.
#define ZN_CLASS_NAME_C(klass)                                                                                         \
	[]() {                                                                                                             \
		static_assert(sizeof(klass) > 0);                                                                              \
		return #klass;                                                                                                 \
	}()

// Gets a method name as a C-string with static lifetime, and causes a compiling error if either the class or method
// doesn't exist.
#define ZN_METHOD_NAME_C(klass, method)                                                                                \
	[]() {                                                                                                             \
		static_assert(sizeof(&klass::method != nullptr));                                                              \
		return #method;                                                                                                \
	}()
