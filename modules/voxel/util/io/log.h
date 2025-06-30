/**************************************************************************/
/*  log.h                                                                 */
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

#include "../string/fwd_std_string.h"

// print_verbose() is used everywhere in Godot, but its drawback is that even if you turn it off, strings
// you print are still allocated and formatted, to not be used. This macro avoids the string.
#define ZN_PRINT_VERBOSE(msg)                                                                                          \
	if (zylann::is_verbose_output_enabled()) {                                                                         \
		zylann::print_line(msg);                                                                                       \
	}

#define ZN_PRINT_WARNING(msg) zylann::print_warning(msg, __FUNCTION__, __FILE__, __LINE__)
#define ZN_PRINT_ERROR(msg) zylann::print_error(msg, __FUNCTION__, __FILE__, __LINE__)

#define ZN_DO_ONCE(stuff)                                                                                              \
	{                                                                                                                  \
		static bool s_first = true;                                                                                    \
		if (s_first) {                                                                                                 \
			s_first = false;                                                                                           \
			stuff;                                                                                                     \
		}                                                                                                              \
	}

#define ZN_PRINT_WARNING_ONCE(msg) ZN_DO_ONCE(ZN_PRINT_WARNING(msg));
#define ZN_PRINT_ERROR_ONCE(msg) ZN_DO_ONCE(ZN_PRINT_ERROR(msg));

namespace zylann {

bool is_verbose_output_enabled();

void print_line(const char *cstr);
void print_line(const FwdConstStdString &s);

void print_warning(const char *warning, const char *func, const char *file, int line);
void print_warning(const FwdConstStdString &warning, const char *func, const char *file, int line);

void print_error(FwdConstStdString error, const char *func, const char *file, int line);
void print_error(const char *error, const char *func, const char *file, int line);
void print_error(const char *error, const char *msg, const char *func, const char *file, int line);
void print_error(const char *error, const FwdConstStdString &msg, const char *func, const char *file, int line);

void flush_stdout();

// When defined, redirects `println` to a file instead of standard output.
// #define ZN_DEBUG_LOG_FILE_ENABLED

#ifdef ZN_DEBUG_LOG_FILE_ENABLED

void open_log_file();
void close_log_file();
void flush_log_file();

#endif

} // namespace zylann
