/**************************************************************************/
/*  otel_log.h                                                           */
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

#ifndef OTEL_LOG_H
#define OTEL_LOG_H

#include "core/io/resource.h"
#include "core/variant/dictionary.h"

class OTelLog : public Resource {
	GDCLASS(OTelLog, Resource);
	friend class OTelDocument;
	friend class OTelState;

public:
	enum SeverityNumber {
		SEVERITY_NUMBER_UNSPECIFIED = 0,
		SEVERITY_NUMBER_TRACE = 1,
		SEVERITY_NUMBER_DEBUG = 5,
		SEVERITY_NUMBER_INFO = 9,
		SEVERITY_NUMBER_WARN = 13,
		SEVERITY_NUMBER_ERROR = 17,
		SEVERITY_NUMBER_FATAL = 21
	};

private:
	uint64_t time_unix_nano = 0;
	uint64_t observed_time_unix_nano = 0;
	SeverityNumber severity_number = SEVERITY_NUMBER_UNSPECIFIED;
	String severity_text;
	String body;
	Dictionary attributes;
	String trace_id;
	String span_id;

protected:
	static void _bind_methods();

public:
	OTelLog();

	// Timestamps
	uint64_t get_time_unix_nano() const;
	void set_time_unix_nano(uint64_t p_time);

	uint64_t get_observed_time_unix_nano() const;
	void set_observed_time_unix_nano(uint64_t p_time);

	// Severity
	SeverityNumber get_severity_number() const;
	void set_severity_number(SeverityNumber p_severity);

	String get_severity_text() const;
	void set_severity_text(const String &p_text);

	// Body
	String get_body() const;
	void set_body(const String &p_body);

	// Attributes
	Dictionary get_attributes() const;
	void set_attributes(const Dictionary &p_attributes);
	void add_attribute(const String &p_key, const Variant &p_value);

	// Trace context
	String get_trace_id() const;
	void set_trace_id(const String &p_trace_id);

	String get_span_id() const;
	void set_span_id(const String &p_span_id);

	// Serialization
	Dictionary to_otlp_dict() const;
	static Ref<OTelLog> from_otlp_dict(const Dictionary &p_dict);
};

VARIANT_ENUM_CAST(OTelLog::SeverityNumber);

#endif // OTEL_LOG_H
