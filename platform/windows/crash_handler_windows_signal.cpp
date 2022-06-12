/*************************************************************************/
/*  crash_handler_windows_signal.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "crash_handler_windows.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

#ifdef CRASH_HANDLER_EXCEPTION

#include <cxxabi.h>
#include <signal.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "thirdparty/libbacktrace/backtrace.h"

int trace_callback(void *data, uintptr_t pc, const char *filename, int lineno, const char *function) {
	if (!function) {
		return 0; // Skip.
	}
	int64_t *idx = reinterpret_cast<int64_t *>(data);

	char fname[1024];
	snprintf(fname, 1024, "%s", function);

	if (function[0] == '_') {
		int status;
		char *demangled = abi::__cxa_demangle(function, nullptr, nullptr, &status);

		if (status == 0 && demangled) {
			snprintf(fname, 1024, "%s", demangled);
		}

		if (demangled) {
			free(demangled);
		}
	}

	print_error(vformat("[%d] %s (%s:%d)", (*idx)++, String::utf8(fname), String::utf8(filename), lineno));
	return 0;
}

void error_callback(void *data, const char *msg, int errnum) {
	int64_t *idx = reinterpret_cast<int64_t *>(data);
	if (*idx == 0) {
		print_error(vformat("Error(%d): %s", errnum, String::utf8(msg)));
	} else {
		print_error(vformat("[%d] error(%d): %s", (*idx)++, errnum, String::utf8(msg)));
	}
}

extern void CrashHandlerException(int signal) {
	if (OS::get_singleton() == nullptr || OS::get_singleton()->is_disable_crash_handler() || IsDebuggerPresent()) {
		return;
	}

	String msg;
	const ProjectSettings *proj_settings = ProjectSettings::get_singleton();
	if (proj_settings) {
		msg = proj_settings->get("debug/settings/crash_handler/message");
	}

	// Tell MainLoop about the crash. This can be handled by users too in Node.
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	print_error("\n================================================================");
	print_error(vformat("%s: Program crashed with signal %d", __FUNCTION__, signal));

	// Print the engine version just before, so that people are reminded to include the version in backtrace reports.
	if (String(VERSION_HASH).is_empty()) {
		print_error(vformat("Engine version: %s", VERSION_FULL_NAME));
	} else {
		print_error(vformat("Engine version: %s (%s)", VERSION_FULL_NAME, VERSION_HASH));
	}
	print_error(vformat("Dumping the backtrace. %s", msg));

	String _execpath = OS::get_singleton()->get_executable_path();

	int64_t idx = 0;
	backtrace_state *state = backtrace_create_state(_execpath.utf8().get_data(), 0, &error_callback, reinterpret_cast<void *>(&idx));
	if (state != nullptr) {
		idx = 1;
		backtrace_full(state, 1, &trace_callback, &error_callback, reinterpret_cast<void *>(&idx));
	}

	print_error("-- END OF BACKTRACE --");
	print_error("================================================================");
}
#endif

CrashHandler::CrashHandler() {
	disabled = false;
}

CrashHandler::~CrashHandler() {
}

void CrashHandler::disable() {
	if (disabled) {
		return;
	}

#if defined(CRASH_HANDLER_EXCEPTION)
	signal(SIGSEGV, nullptr);
	signal(SIGFPE, nullptr);
	signal(SIGILL, nullptr);
#endif

	disabled = true;
}

void CrashHandler::initialize() {
#if defined(CRASH_HANDLER_EXCEPTION)
	signal(SIGSEGV, CrashHandlerException);
	signal(SIGFPE, CrashHandlerException);
	signal(SIGILL, CrashHandlerException);
#endif
}
