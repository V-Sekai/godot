/**************************************************************************/
/*  errors.h                                                              */
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

#include "io/log.h"
#include "macros.h"

// Abnormally terminate the program
#ifdef _MSC_VER
#define ZN_GENERATE_TRAP() __debugbreak()
#else
#define ZN_GENERATE_TRAP() __builtin_trap()
#endif

// The following macros print a message and crash the program.

#define ZN_CRASH_MSG(msg)                                                                                              \
	zylann::print_error("FATAL: Method/function failed.", msg, __FUNCTION__, __FILE__, __LINE__);                      \
	zylann::flush_stdout();                                                                                            \
	ZN_GENERATE_TRAP()

#define ZN_CRASH() ZN_CRASH_MSG("")

// The following macros check a condition. If it fails, they print a message and crash the program.

#define ZN_ASSERT_MSG(cond, msg)                                                                                       \
	if (ZN_UNLIKELY(!(cond))) {                                                                                        \
		zylann::print_error(                                                                                           \
				"FATAL: Assertion failed: \"" #cond "\" is false.", msg, __FUNCTION__, __FILE__, __LINE__              \
		);                                                                                                             \
		zylann::flush_stdout();                                                                                        \
		ZN_GENERATE_TRAP();                                                                                            \
	} else                                                                                                             \
		((void)0)

#define ZN_ASSERT(cond) ZN_ASSERT_MSG(cond, "")

// The following macros check a condition. If it fails, they print a message, then return or continue.

#define ZN_INTERNAL_ASSERT_ACT(cond, act, msg)                                                                         \
	if (ZN_UNLIKELY(!(cond))) {                                                                                        \
		zylann::print_error("Assertion failed: \"" #cond "\" is false.", msg, __FUNCTION__, __FILE__, __LINE__);       \
		act;                                                                                                           \
	} else                                                                                                             \
		((void)0)

#define ZN_ASSERT_RETURN(cond) ZN_INTERNAL_ASSERT_ACT(cond, return, "")
#define ZN_ASSERT_RETURN_MSG(cond, msg) ZN_INTERNAL_ASSERT_ACT(cond, return, msg)
#define ZN_ASSERT_RETURN_V(cond, retval) ZN_INTERNAL_ASSERT_ACT(cond, return retval, "")
#define ZN_ASSERT_RETURN_V_MSG(cond, retval, msg) ZN_INTERNAL_ASSERT_ACT(cond, return retval, msg)
#define ZN_ASSERT_CONTINUE(cond) ZN_INTERNAL_ASSERT_ACT(cond, continue, "")
#define ZN_ASSERT_CONTINUE_MSG(cond, msg) ZN_INTERNAL_ASSERT_ACT(cond, continue, msg)
