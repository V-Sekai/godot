/**************************************************************************/
/*  open_telemetry_logger.cpp                                             */
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

#include "open_telemetry_logger.h"

#include "core/os/time.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"

OpenTelemetryLogger::OpenTelemetryLogger(OpenTelemetry *p_otel) {
	otel_instance = p_otel;
	enabled = true;
}

OpenTelemetryLogger::~OpenTelemetryLogger() {
	// Don't delete otel_instance, we don't own it
}

void OpenTelemetryLogger::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!enabled || !otel_instance) {
		return;
	}

	// Avoid infinite recursion - don't log OpenTelemetry's own messages
	// Use thread_local for thread safety
	thread_local static bool logging_in_progress = false;
	if (logging_in_progress) {
		return;
	}

	logging_in_progress = true;

	// Format the message using Godot's vformat
	String message = vformat(p_format, p_list);

	// Map to log level
	String level = p_err ? "ERROR" : "INFO";

	// Build attributes
	Dictionary attrs;
	attrs["source"] = "godot_stdout";
	attrs["error_stream"] = p_err;

	// Send to OpenTelemetry
	otel_instance->log_message(level, message, attrs);

	logging_in_progress = false;
}

void OpenTelemetryLogger::log_error(const char *p_function, const char *p_file, int p_line,
		const char *p_code, const char *p_rationale,
		bool p_editor_notify, ErrorType p_type,
		const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!enabled || !otel_instance) {
		return;
	}

	// Avoid infinite recursion - use thread_local for thread safety
	thread_local static bool logging_in_progress = false;
	if (logging_in_progress) {
		return;
	}

	logging_in_progress = true;

	// Map error type to log level
	String level;
	switch (p_type) {
		case ERR_ERROR:
			level = "ERROR";
			break;
		case ERR_WARNING:
			level = "WARN";
			break;
		case ERR_SCRIPT:
			level = "ERROR";
			break;
		case ERR_SHADER:
			level = "ERROR";
			break;
		default:
			level = "ERROR";
			break;
	}

	// Build detailed attributes
	Dictionary attrs;
	attrs["source"] = "godot_error";
	attrs["function"] = String(p_function);
	attrs["file"] = String(p_file);
	attrs["line"] = p_line;
	attrs["code"] = String(p_code);
	attrs["error_type"] = error_type_string(p_type);
	attrs["editor_notify"] = p_editor_notify;

	// Add script backtrace if available
	if (p_script_backtraces.size() > 0) {
		String backtrace_str;
		for (int i = 0; i < p_script_backtraces.size(); i++) {
			Ref<ScriptBacktrace> bt = p_script_backtraces[i];
			if (bt.is_valid() && bt->get_frame_count() > 0) {
				// Add each frame in the backtrace
				for (int j = 0; j < bt->get_frame_count(); j++) {
					if (i > 0 || j > 0) {
						backtrace_str += "\n";
					}
					backtrace_str += bt->get_frame_file(j) + ":" + itos(bt->get_frame_line(j)) + " @ " + bt->get_frame_function(j) + "()";
				}
			}
		}
		if (!backtrace_str.is_empty()) {
			attrs["backtrace"] = backtrace_str;
		}
	}

	// Send to OpenTelemetry
	String message = String(p_rationale);
	otel_instance->log_message(level, message, attrs);

	logging_in_progress = false;
}
