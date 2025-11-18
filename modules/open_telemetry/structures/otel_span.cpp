/**************************************************************************/
/*  otel_span.cpp                                                         */
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

#include "otel_span.h"

#include "core/crypto/crypto.h"
#include "core/os/time.h"
#include "core/string/string_builder.h"

void OpenTelemetrySpan::_bind_methods() {
	// Validation methods
	ClassDB::bind_static_method("OpenTelemetrySpan", D_METHOD("is_valid_trace_id", "id"), &OpenTelemetrySpan::is_valid_trace_id);
	ClassDB::bind_static_method("OpenTelemetrySpan", D_METHOD("is_valid_span_id", "id"), &OpenTelemetrySpan::is_valid_span_id);
	ClassDB::bind_static_method("OpenTelemetrySpan", D_METHOD("generate_trace_id"), &OpenTelemetrySpan::generate_trace_id);
	ClassDB::bind_static_method("OpenTelemetrySpan", D_METHOD("generate_span_id"), &OpenTelemetrySpan::generate_span_id);

	// Trace and Span IDs
	ClassDB::bind_method(D_METHOD("get_trace_id"), &OpenTelemetrySpan::get_trace_id);
	ClassDB::bind_method(D_METHOD("set_trace_id", "trace_id"), &OpenTelemetrySpan::set_trace_id);
	ClassDB::bind_method(D_METHOD("get_span_id"), &OpenTelemetrySpan::get_span_id);
	ClassDB::bind_method(D_METHOD("set_span_id", "span_id"), &OpenTelemetrySpan::set_span_id);
	ClassDB::bind_method(D_METHOD("get_parent_span_id"), &OpenTelemetrySpan::get_parent_span_id);
	ClassDB::bind_method(D_METHOD("set_parent_span_id", "parent_span_id"), &OpenTelemetrySpan::set_parent_span_id);

	// Span metadata
	// Note: get_name() and set_name() are inherited from Resource, so we don't bind them here
	ClassDB::bind_method(D_METHOD("get_kind"), &OpenTelemetrySpan::get_kind);
	ClassDB::bind_method(D_METHOD("set_kind", "kind"), &OpenTelemetrySpan::set_kind);

	// Timestamps
	ClassDB::bind_method(D_METHOD("get_start_time_unix_nano"), &OpenTelemetrySpan::get_start_time_unix_nano);
	ClassDB::bind_method(D_METHOD("set_start_time_unix_nano", "time"), &OpenTelemetrySpan::set_start_time_unix_nano);
	ClassDB::bind_method(D_METHOD("get_end_time_unix_nano"), &OpenTelemetrySpan::get_end_time_unix_nano);
	ClassDB::bind_method(D_METHOD("set_end_time_unix_nano", "time"), &OpenTelemetrySpan::set_end_time_unix_nano);

	// Attributes
	ClassDB::bind_method(D_METHOD("get_attributes"), &OpenTelemetrySpan::get_attributes);
	ClassDB::bind_method(D_METHOD("set_attributes", "attributes"), &OpenTelemetrySpan::set_attributes);
	ClassDB::bind_method(D_METHOD("add_attribute", "key", "value"), &OpenTelemetrySpan::add_attribute);

	// Events
	ClassDB::bind_method(D_METHOD("get_events"), &OpenTelemetrySpan::get_events);
	ClassDB::bind_method(D_METHOD("set_events", "events"), &OpenTelemetrySpan::set_events);
	ClassDB::bind_method(D_METHOD("add_event", "name", "attributes", "timestamp"), &OpenTelemetrySpan::add_event, DEFVAL(Dictionary()), DEFVAL(0));

	// Links
	ClassDB::bind_method(D_METHOD("get_links"), &OpenTelemetrySpan::get_links);
	ClassDB::bind_method(D_METHOD("set_links", "links"), &OpenTelemetrySpan::set_links);
	ClassDB::bind_method(D_METHOD("add_link", "trace_id", "span_id", "attributes"), &OpenTelemetrySpan::add_link, DEFVAL(Dictionary()));

	// Status
	ClassDB::bind_method(D_METHOD("get_status_code"), &OpenTelemetrySpan::get_status_code);
	ClassDB::bind_method(D_METHOD("set_status_code", "code"), &OpenTelemetrySpan::set_status_code);
	ClassDB::bind_method(D_METHOD("get_status_message"), &OpenTelemetrySpan::get_status_message);
	ClassDB::bind_method(D_METHOD("set_status_message", "message"), &OpenTelemetrySpan::set_status_message);

	// State
	ClassDB::bind_method(D_METHOD("is_ended"), &OpenTelemetrySpan::is_ended);
	ClassDB::bind_method(D_METHOD("mark_ended"), &OpenTelemetrySpan::mark_ended);

	// Serialization
	ClassDB::bind_method(D_METHOD("to_otlp_dict"), &OpenTelemetrySpan::to_otlp_dict);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "trace_id"), "set_trace_id", "get_trace_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "span_id"), "set_span_id", "get_span_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "parent_span_id"), "set_parent_span_id", "get_parent_span_id");
	// Note: "name" property is inherited from Resource, so we don't add it here
	ADD_PROPERTY(PropertyInfo(Variant::INT, "kind", PROPERTY_HINT_ENUM, "Unspecified,Internal,Server,Client,Producer,Consumer"), "set_kind", "get_kind");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "start_time_unix_nano"), "set_start_time_unix_nano", "get_start_time_unix_nano");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "end_time_unix_nano"), "set_end_time_unix_nano", "get_end_time_unix_nano");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "attributes"), "set_attributes", "get_attributes");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "events"), "set_events", "get_events");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "links"), "set_links", "get_links");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "status_code", PROPERTY_HINT_ENUM, "Unset,Ok,Error"), "set_status_code", "get_status_code");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "status_message"), "set_status_message", "get_status_message");

	BIND_ENUM_CONSTANT(SPAN_KIND_UNSPECIFIED);
	BIND_ENUM_CONSTANT(SPAN_KIND_INTERNAL);
	BIND_ENUM_CONSTANT(SPAN_KIND_SERVER);
	BIND_ENUM_CONSTANT(SPAN_KIND_CLIENT);
	BIND_ENUM_CONSTANT(SPAN_KIND_PRODUCER);
	BIND_ENUM_CONSTANT(SPAN_KIND_CONSUMER);

	BIND_ENUM_CONSTANT(STATUS_CODE_UNSET);
	BIND_ENUM_CONSTANT(STATUS_CODE_OK);
	BIND_ENUM_CONSTANT(STATUS_CODE_ERROR);
}

OpenTelemetrySpan::OpenTelemetrySpan() {
	// Initialize with empty IDs (will be generated when needed)
	// Don't generate random IDs here to ensure consistent default values for documentation
	trace_id = "";
	span_id = "";
}

// Validation helpers (private)
bool OpenTelemetrySpan::is_valid_hex_string(const String &p_str, int p_expected_length) {
	if (p_str.length() != p_expected_length) {
		return false;
	}

	for (int i = 0; i < p_str.length(); i++) {
		char32_t c = p_str[i];
		bool is_hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
		if (!is_hex) {
			return false;
		}
	}

	return true;
}

String OpenTelemetrySpan::generate_random_hex(int p_length) {
	// Generate cryptographically secure random hex string
	Ref<Crypto> crypto = Crypto::create();
	int byte_count = (p_length + 1) / 2;
	PackedByteArray random_bytes = crypto->generate_random_bytes(byte_count);

	StringBuilder hex;
	for (int i = 0; i < random_bytes.size(); i++) {
		hex.append(String::num_int64(random_bytes[i], 16).pad_zeros(2));
	}

	return hex.as_string().substr(0, p_length).to_lower();
}

// Validation methods (public)
bool OpenTelemetrySpan::is_valid_trace_id(const String &p_id) {
	// Allow empty (unset)
	if (p_id.is_empty()) {
		return true;
	}
	// Must be exactly 32 hex characters
	return is_valid_hex_string(p_id, 32);
}

bool OpenTelemetrySpan::is_valid_span_id(const String &p_id) {
	// Allow empty (unset)
	if (p_id.is_empty()) {
		return true;
	}
	// Must be exactly 16 hex characters
	return is_valid_hex_string(p_id, 16);
}

// ID generation helpers
String OpenTelemetrySpan::generate_trace_id() {
	// Generate UUIDv7 as 32 hex characters (128 bits)
	// UUIDv7 format: unix_ts_ms(48) + ver(4) + rand_a(12) + var(2) + rand_b(62)

	Ref<Crypto> crypto = Crypto::create();

	// Get current timestamp in milliseconds
	uint64_t timestamp_ms = Time::get_singleton()->get_unix_time_from_system() * 1000;

	// Generate 10 bytes of cryptographically secure random data
	PackedByteArray random_bytes = crypto->generate_random_bytes(10);

	// Build UUIDv7 structure (128 bits / 32 hex chars)
	// Bits 0-47: timestamp_ms
	uint64_t time_hi = (timestamp_ms >> 16) & 0xFFFFFFFF; // Upper 32 bits
	uint64_t time_lo = (timestamp_ms & 0xFFFF); // Lower 16 bits

	// Bits 48-63: version (0111 = 7) + 12 random bits
	uint16_t rand_12bits = ((uint16_t)random_bytes[0] << 4) | ((random_bytes[1] >> 4) & 0xF);
	uint64_t ver_rand = (0x7ULL << 12) | rand_12bits;

	// Bits 64-127: variant (10) + 62 random bits
	uint64_t rand_62bits = ((uint64_t)(random_bytes[1] & 0xF) << 58) |
			((uint64_t)random_bytes[2] << 50) |
			((uint64_t)random_bytes[3] << 42) |
			((uint64_t)random_bytes[4] << 34) |
			((uint64_t)random_bytes[5] << 26) |
			((uint64_t)random_bytes[6] << 18) |
			((uint64_t)random_bytes[7] << 10) |
			((uint64_t)random_bytes[8] << 2) |
			((random_bytes[9] >> 6) & 0x3);

	uint64_t var_hi = (0x2ULL << 30) | ((rand_62bits >> 32) & 0x3FFFFFFF);
	uint64_t var_lo = rand_62bits & 0xFFFFFFFF;

	// Convert to 32-character hex string
	// Each part must be exactly 8 hex characters
	// pad_zeros only pads if shorter, so we need to truncate if longer
	StringBuilder uuid;
	String part1 = String::num_int64(time_hi, 16).pad_zeros(8);
	if (part1.length() > 8) {
		part1 = part1.substr(part1.length() - 8, 8); // Take last 8 chars
	}
	uuid.append(part1);

	String part2 = String::num_int64((time_lo << 16) | ver_rand, 16).pad_zeros(8);
	if (part2.length() > 8) {
		part2 = part2.substr(part2.length() - 8, 8); // Take last 8 chars
	}
	uuid.append(part2);

	String part3 = String::num_int64(var_hi, 16).pad_zeros(8);
	if (part3.length() > 8) {
		part3 = part3.substr(part3.length() - 8, 8); // Take last 8 chars
	}
	uuid.append(part3);

	String part4 = String::num_int64(var_lo, 16).pad_zeros(8);
	if (part4.length() > 8) {
		part4 = part4.substr(part4.length() - 8, 8); // Take last 8 chars
	}
	uuid.append(part4);

	return uuid.as_string().to_lower();
}

String OpenTelemetrySpan::generate_span_id() {
	return generate_random_hex(16);
}

// Trace and Span IDs
String OpenTelemetrySpan::get_trace_id() const {
	return trace_id;
}

void OpenTelemetrySpan::set_trace_id(const String &p_trace_id) {
	ERR_FAIL_COND_MSG(!is_valid_trace_id(p_trace_id),
			vformat("Invalid trace_id: '%s'. Must be exactly 32 hexadecimal characters or empty.", p_trace_id));
	trace_id = p_trace_id;
}

String OpenTelemetrySpan::get_span_id() const {
	return span_id;
}

void OpenTelemetrySpan::set_span_id(const String &p_span_id) {
	ERR_FAIL_COND_MSG(!is_valid_span_id(p_span_id),
			vformat("Invalid span_id: '%s'. Must be exactly 16 hexadecimal characters or empty.", p_span_id));
	span_id = p_span_id;
}

String OpenTelemetrySpan::get_parent_span_id() const {
	return parent_span_id;
}

void OpenTelemetrySpan::set_parent_span_id(const String &p_parent_span_id) {
	ERR_FAIL_COND_MSG(!is_valid_span_id(p_parent_span_id),
			vformat("Invalid parent_span_id: '%s'. Must be exactly 16 hexadecimal characters or empty.", p_parent_span_id));
	parent_span_id = p_parent_span_id;
}

// Span metadata
String OpenTelemetrySpan::get_name() const {
	// Use Resource's built-in name property
	return Resource::get_name();
}

void OpenTelemetrySpan::set_name(const String &p_name) {
	// Use Resource's built-in name property
	Resource::set_name(p_name);
}

OpenTelemetrySpan::SpanKind OpenTelemetrySpan::get_kind() const {
	return kind;
}

void OpenTelemetrySpan::set_kind(SpanKind p_kind) {
	kind = p_kind;
}

// Timestamps
uint64_t OpenTelemetrySpan::get_start_time_unix_nano() const {
	return start_time_unix_nano;
}

void OpenTelemetrySpan::set_start_time_unix_nano(uint64_t p_time) {
	start_time_unix_nano = p_time;
}

uint64_t OpenTelemetrySpan::get_end_time_unix_nano() const {
	return end_time_unix_nano;
}

void OpenTelemetrySpan::set_end_time_unix_nano(uint64_t p_time) {
	end_time_unix_nano = p_time;
}

// Attributes
Dictionary OpenTelemetrySpan::get_attributes() const {
	return attributes;
}

void OpenTelemetrySpan::set_attributes(const Dictionary &p_attributes) {
	attributes = p_attributes;
}

void OpenTelemetrySpan::add_attribute(const String &p_key, const Variant &p_value) {
	attributes[p_key] = p_value;
}

// Events
TypedArray<Dictionary> OpenTelemetrySpan::get_events() const {
	return events;
}

void OpenTelemetrySpan::set_events(const TypedArray<Dictionary> &p_events) {
	events = p_events;
}

void OpenTelemetrySpan::add_event(const String &p_name, const Dictionary &p_attributes, uint64_t p_timestamp) {
	Dictionary event;
	event["name"] = p_name;
	event["timeUnixNano"] = p_timestamp == 0 ? (uint64_t)(Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL) : p_timestamp;
	if (p_attributes.size() > 0) {
		event["attributes"] = p_attributes;
	}
	events.push_back(event);
}

// Links
TypedArray<Dictionary> OpenTelemetrySpan::get_links() const {
	return links;
}

void OpenTelemetrySpan::set_links(const TypedArray<Dictionary> &p_links) {
	links = p_links;
}

void OpenTelemetrySpan::add_link(const String &p_trace_id, const String &p_span_id, const Dictionary &p_attributes) {
	ERR_FAIL_COND_MSG(!is_valid_trace_id(p_trace_id),
			vformat("Invalid link trace_id: '%s'. Must be exactly 32 hexadecimal characters.", p_trace_id));
	ERR_FAIL_COND_MSG(!is_valid_span_id(p_span_id),
			vformat("Invalid link span_id: '%s'. Must be exactly 16 hexadecimal characters.", p_span_id));

	Dictionary link;
	link["traceId"] = p_trace_id;
	link["spanId"] = p_span_id;
	if (p_attributes.size() > 0) {
		link["attributes"] = p_attributes;
	}
	links.push_back(link);
}

// Status
OpenTelemetrySpan::StatusCode OpenTelemetrySpan::get_status_code() const {
	return status_code;
}

void OpenTelemetrySpan::set_status_code(StatusCode p_code) {
	status_code = p_code;
}

String OpenTelemetrySpan::get_status_message() const {
	return status_message;
}

void OpenTelemetrySpan::set_status_message(const String &p_message) {
	status_message = p_message;
}

// State
bool OpenTelemetrySpan::is_ended() const {
	return ended;
}

void OpenTelemetrySpan::mark_ended() {
	if (!ended) {
		ended = true;
		if (end_time_unix_nano == 0) {
			end_time_unix_nano = (uint64_t)(Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL);
		}
	}
}

// Serialization
Dictionary OpenTelemetrySpan::to_otlp_dict() const {
	Dictionary span_dict;

	span_dict["traceId"] = trace_id;
	span_dict["spanId"] = span_id;

	if (!parent_span_id.is_empty()) {
		span_dict["parentSpanId"] = parent_span_id;
	}

	span_dict["name"] = get_name();
	span_dict["kind"] = (int)kind;
	span_dict["startTimeUnixNano"] = (int64_t)start_time_unix_nano;
	span_dict["endTimeUnixNano"] = (int64_t)end_time_unix_nano;

	if (attributes.size() > 0) {
		span_dict["attributes"] = attributes;
	}

	if (events.size() > 0) {
		span_dict["events"] = events;
	}

	if (links.size() > 0) {
		span_dict["links"] = links;
	}

	if (status_code != STATUS_CODE_UNSET) {
		Dictionary status;
		status["code"] = (int)status_code;
		if (!status_message.is_empty()) {
			status["message"] = status_message;
		}
		span_dict["status"] = status;
	}

	return span_dict;
}

Ref<OpenTelemetrySpan> OpenTelemetrySpan::from_otlp_dict(const Dictionary &p_dict) {
	Ref<OpenTelemetrySpan> span;
	span.instantiate();

	if (p_dict.has("traceId")) {
		span->set_trace_id(p_dict["traceId"]);
	}
	if (p_dict.has("spanId")) {
		span->set_span_id(p_dict["spanId"]);
	}
	if (p_dict.has("parentSpanId")) {
		span->set_parent_span_id(p_dict["parentSpanId"]);
	}
	if (p_dict.has("name")) {
		span->set_name(p_dict["name"]);
	}
	if (p_dict.has("kind")) {
		span->set_kind((SpanKind)(int)p_dict["kind"]);
	}
	if (p_dict.has("startTimeUnixNano")) {
		span->set_start_time_unix_nano((uint64_t)(int64_t)p_dict["startTimeUnixNano"]);
	}
	if (p_dict.has("endTimeUnixNano")) {
		span->set_end_time_unix_nano((uint64_t)(int64_t)p_dict["endTimeUnixNano"]);
	}
	if (p_dict.has("attributes")) {
		span->set_attributes(p_dict["attributes"]);
	}
	if (p_dict.has("events")) {
		span->set_events(p_dict["events"]);
	}
	if (p_dict.has("links")) {
		span->set_links(p_dict["links"]);
	}
	if (p_dict.has("status")) {
		Dictionary status = p_dict["status"];
		if (status.has("code")) {
			span->set_status_code((StatusCode)(int)status["code"]);
		}
		if (status.has("message")) {
			span->set_status_message(status["message"]);
		}
	}

	return span;
}
