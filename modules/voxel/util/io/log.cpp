/**************************************************************************/
/*  log.cpp                                                               */
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

#include "../godot/classes/os.h"
#include "../godot/core/print_string.h"
#include "../string/format.h"

#ifdef ZN_DEBUG_LOG_FILE_ENABLED
#include "thread/mutex.h"
#include <fstream>
#endif

namespace zylann {

#ifdef ZN_DEBUG_LOG_FILE_ENABLED

namespace {
Mutex g_log_file_mutex;
bool g_log_to_file = false;
std::ofstream g_log_ofs;
} // namespace

void open_log_file() {
	MutexLock mlock(g_log_file_mutex);
	g_log_to_file = true;
	g_log_ofs.open("zn_log.txt", std::ios::binary | std::ios::trunc);
}

void close_log_file() {
	MutexLock mlock(g_log_file_mutex);
	g_log_to_file = false;
	g_log_ofs.close();
}

void flush_log_file() {
	MutexLock mlock(g_log_file_mutex);
	if (g_log_to_file) {
		g_log_ofs.flush();
	}
}

#endif

bool is_verbose_output_enabled() {
	return OS::get_singleton()->is_stdout_verbose();
}

void print_line(const char *cstr) {
#ifdef ZN_DEBUG_LOG_FILE_ENABLED
	if (g_log_to_file) {
		MutexLock mlock(g_log_file_mutex);
		if (g_log_to_file) {
			g_log_ofs.write(cstr, strlen(cstr));
			g_log_ofs.write("\n", 1);
		}
	}
#else

#if defined(ZN_GODOT)
	::print_line(cstr);
#elif defined(ZN_GODOT_EXTENSION)
	::godot::UtilityFunctions::print(cstr);
#endif

#endif
}

void print_line(const FwdConstStdString &s) {
	print_line(s.s.c_str());
}

void print_warning(const char *message, const char *func, const char *file, int line) {
#if defined(ZN_GODOT)
	_err_print_error(func, file, line, message, false, ERR_HANDLER_WARNING);
#elif defined(ZN_GODOT_EXTENSION)
	::godot::_err_print_error(func, file, line, message, false, true);
#endif
}

void print_warning(const FwdConstStdString &warning, const char *func, const char *file, int line) {
	print_warning(warning.s.c_str(), func, file, line);
}

void print_error(FwdConstStdString error, const char *func, const char *file, int line) {
	_err_print_error(func, file, line, error.s.c_str());
}

void print_error(const char *error, const char *func, const char *file, int line) {
	_err_print_error(func, file, line, error);
}

void print_error(const char *error, const char *msg, const char *func, const char *file, int line) {
	_err_print_error(func, file, line, error, msg);
}

void print_error(const char *error, const FwdConstStdString &msg, const char *func, const char *file, int line) {
	_err_print_error(func, file, line, error, msg.s.c_str());
}

void flush_stdout() {
	_err_flush_stdout();
}

} // namespace zylann
