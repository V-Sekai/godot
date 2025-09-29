/**************************************************************************/
/*  otel_span.h                                                          */
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

#ifndef OTEL_SPAN_H
#define OTEL_SPAN_H

#include "core/io/resource.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

class OTelSpan : public Resource {
	GDCLASS(OTelSpan, Resource);
	friend class OTelDocument;
	friend class OTelState;

public:
	enum SpanKind {
		SPAN_KIND_UNSPECIFIED = 0,
		SPAN_KIND_INTERNAL = 1,
		SPAN_KIND_SERVER = 2,
		SPAN_KIND_CLIENT = 3,
		SPAN_KIND_PRODUCER = 4,
		SPAN_KIND_CONSUMER = 5
	};

	enum StatusCode {
		STATUS_CODE_UNSET = 0,
		STATUS_CODE_OK = 1,
		STATUS_CODE_ERROR = 2
	};

private:
	// Core span data
	String trace_id; // 32 hex characters (128 bits)
	String span_id; // 16 hex characters (64 bits)
	String parent_span_id; // Empty if root span
	String name;
	SpanKind kind = SPAN_KIND_INTERNAL;

	// Validation helpers
	static bool is_valid_hex_string(const String &p_str, int p_expected_length);
	static String generate_random_hex(int p_length);

	// Timestamps (nanoseconds since Unix epoch)
	uint64_t start_time_unix_nano = 0;
	uint64_t end_time_unix_nano = 0;

	// Attributes (key-value pairs)
	Dictionary attributes;

	// Events
	TypedArray<Dictionary> events;

	// Links to other spans
	TypedArray<Dictionary> links;

	// Status
	StatusCode status_code = STATUS_CODE_UNSET;
	String status_message;

	// Flags
	bool ended = false;

protected:
	static void _bind_methods();

public:
	OTelSpan();

	// Validation methods
	static bool is_valid_trace_id(const String &p_id);
	static bool is_valid_span_id(const String &p_id);
	
	// ID generation helpers
	static String generate_trace_id();
	static String generate_span_id();

	// Trace and Span IDs
	String get_trace_id() const;
	void set_trace_id(const String &p_trace_id);

	String get_span_id() const;
	void set_span_id(const String &p_span_id);

	String get_parent_span_id() const;
	void set_parent_span_id(const String &p_parent_span_id);

	// Span metadata
	String get_name() const;
	void set_name(const String &p_name);

	SpanKind get_kind() const;
	void set_kind(SpanKind p_kind);

	// Timestamps
	uint64_t get_start_time_unix_nano() const;
	void set_start_time_unix_nano(uint64_t p_time);

	uint64_t get_end_time_unix_nano() const;
	void set_end_time_unix_nano(uint64_t p_time);

	// Attributes
	Dictionary get_attributes() const;
	void set_attributes(const Dictionary &p_attributes);
	void add_attribute(const String &p_key, const Variant &p_value);

	// Events
	TypedArray<Dictionary> get_events() const;
	void set_events(const TypedArray<Dictionary> &p_events);
	void add_event(const String &p_name, const Dictionary &p_attributes = Dictionary(), uint64_t p_timestamp = 0);

	// Links
	TypedArray<Dictionary> get_links() const;
	void set_links(const TypedArray<Dictionary> &p_links);
	void add_link(const String &p_trace_id, const String &p_span_id, const Dictionary &p_attributes = Dictionary());

	// Status
	StatusCode get_status_code() const;
	void set_status_code(StatusCode p_code);

	String get_status_message() const;
	void set_status_message(const String &p_message);

	// State
	bool is_ended() const;
	void mark_ended();

	// Serialization
	Dictionary to_otlp_dict() const;
	static Ref<OTelSpan> from_otlp_dict(const Dictionary &p_dict);
};

VARIANT_ENUM_CAST(OTelSpan::SpanKind);
VARIANT_ENUM_CAST(OTelSpan::StatusCode);

#endif // OTEL_SPAN_H
