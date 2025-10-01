/**************************************************************************/
/*  otel_log.cpp                                                          */
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

#include "otel_log.h"

#include "core/os/time.h"

void OTelLog::_bind_methods() {
	// Timestamps
	ClassDB::bind_method(D_METHOD("get_time_unix_nano"), &OTelLog::get_time_unix_nano);
	ClassDB::bind_method(D_METHOD("set_time_unix_nano", "time"), &OTelLog::set_time_unix_nano);
	ClassDB::bind_method(D_METHOD("get_observed_time_unix_nano"), &OTelLog::get_observed_time_unix_nano);
	ClassDB::bind_method(D_METHOD("set_observed_time_unix_nano", "time"), &OTelLog::set_observed_time_unix_nano);

	// Severity
	ClassDB::bind_method(D_METHOD("get_severity_number"), &OTelLog::get_severity_number);
	ClassDB::bind_method(D_METHOD("set_severity_number", "severity"), &OTelLog::set_severity_number);
	ClassDB::bind_method(D_METHOD("get_severity_text"), &OTelLog::get_severity_text);
	ClassDB::bind_method(D_METHOD("set_severity_text", "text"), &OTelLog::set_severity_text);

	// Body
	ClassDB::bind_method(D_METHOD("get_body"), &OTelLog::get_body);
	ClassDB::bind_method(D_METHOD("set_body", "body"), &OTelLog::set_body);

	// Attributes
	ClassDB::bind_method(D_METHOD("get_attributes"), &OTelLog::get_attributes);
	ClassDB::bind_method(D_METHOD("set_attributes", "attributes"), &OTelLog::set_attributes);
	ClassDB::bind_method(D_METHOD("add_attribute", "key", "value"), &OTelLog::add_attribute);

	// Trace context
	ClassDB::bind_method(D_METHOD("get_trace_id"), &OTelLog::get_trace_id);
	ClassDB::bind_method(D_METHOD("set_trace_id", "trace_id"), &OTelLog::set_trace_id);
	ClassDB::bind_method(D_METHOD("get_span_id"), &OTelLog::get_span_id);
	ClassDB::bind_method(D_METHOD("set_span_id", "span_id"), &OTelLog::set_span_id);

	// Serialization
	ClassDB::bind_method(D_METHOD("to_otlp_dict"), &OTelLog::to_otlp_dict);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "time_unix_nano"), "set_time_unix_nano", "get_time_unix_nano");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "observed_time_unix_nano"), "set_observed_time_unix_nano", "get_observed_time_unix_nano");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "severity_number"), "set_severity_number", "get_severity_number");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "severity_text"), "set_severity_text", "get_severity_text");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "body"), "set_body", "get_body");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "attributes"), "set_attributes", "get_attributes");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "trace_id"), "set_trace_id", "get_trace_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "span_id"), "set_span_id", "get_span_id");

	BIND_ENUM_CONSTANT(SEVERITY_NUMBER_UNSPECIFIED);
	BIND_ENUM_CONSTANT(SEVERITY_NUMBER_TRACE);
	BIND_ENUM_CONSTANT(SEVERITY_NUMBER_DEBUG);
	BIND_ENUM_CONSTANT(SEVERITY_NUMBER_INFO);
	BIND_ENUM_CONSTANT(SEVERITY_NUMBER_WARN);
	BIND_ENUM_CONSTANT(SEVERITY_NUMBER_ERROR);
	BIND_ENUM_CONSTANT(SEVERITY_NUMBER_FATAL);
}

OTelLog::OTelLog() {
}

// Timestamps
uint64_t OTelLog::get_time_unix_nano() const {
	return time_unix_nano;
}

void OTelLog::set_time_unix_nano(uint64_t p_time) {
	time_unix_nano = p_time;
}

uint64_t OTelLog::get_observed_time_unix_nano() const {
	return observed_time_unix_nano;
}

void OTelLog::set_observed_time_unix_nano(uint64_t p_time) {
	observed_time_unix_nano = p_time;
}

// Severity
OTelLog::SeverityNumber OTelLog::get_severity_number() const {
	return severity_number;
}

void OTelLog::set_severity_number(SeverityNumber p_severity) {
	severity_number = p_severity;
}

String OTelLog::get_severity_text() const {
	return severity_text;
}

void OTelLog::set_severity_text(const String &p_text) {
	severity_text = p_text;
}

// Body
String OTelLog::get_body() const {
	return body;
}

void OTelLog::set_body(const String &p_body) {
	body = p_body;
}

// Attributes
Dictionary OTelLog::get_attributes() const {
	return attributes;
}

void OTelLog::set_attributes(const Dictionary &p_attributes) {
	attributes = p_attributes;
}

void OTelLog::add_attribute(const String &p_key, const Variant &p_value) {
	attributes[p_key] = p_value;
}

// Trace context
String OTelLog::get_trace_id() const {
	return trace_id;
}

void OTelLog::set_trace_id(const String &p_trace_id) {
	trace_id = p_trace_id;
}

String OTelLog::get_span_id() const {
	return span_id;
}

void OTelLog::set_span_id(const String &p_span_id) {
	span_id = p_span_id;
}

// Serialization
Dictionary OTelLog::to_otlp_dict() const {
	Dictionary log_dict;

	log_dict["timeUnixNano"] = (int64_t)time_unix_nano;

	if (observed_time_unix_nano > 0) {
		log_dict["observedTimeUnixNano"] = (int64_t)observed_time_unix_nano;
	}

	if (severity_number != SEVERITY_NUMBER_UNSPECIFIED) {
		log_dict["severityNumber"] = (int)severity_number;
	}

	if (!severity_text.is_empty()) {
		log_dict["severityText"] = severity_text;
	}

	if (!body.is_empty()) {
		Dictionary body_value;
		body_value["stringValue"] = body;
		log_dict["body"] = body_value;
	}

	if (attributes.size() > 0) {
		log_dict["attributes"] = attributes;
	}

	if (!trace_id.is_empty()) {
		log_dict["traceId"] = trace_id;
	}

	if (!span_id.is_empty()) {
		log_dict["spanId"] = span_id;
	}

	return log_dict;
}

Ref<OTelLog> OTelLog::from_otlp_dict(const Dictionary &p_dict) {
	Ref<OTelLog> log;
	log.instantiate();

	if (p_dict.has("timeUnixNano")) {
		log->set_time_unix_nano((uint64_t)(int64_t)p_dict["timeUnixNano"]);
	}

	if (p_dict.has("observedTimeUnixNano")) {
		log->set_observed_time_unix_nano((uint64_t)(int64_t)p_dict["observedTimeUnixNano"]);
	}

	if (p_dict.has("severityNumber")) {
		log->set_severity_number((SeverityNumber)(int)p_dict["severityNumber"]);
	}

	if (p_dict.has("severityText")) {
		log->set_severity_text(p_dict["severityText"]);
	}

	if (p_dict.has("body")) {
		Dictionary body_value = p_dict["body"];
		if (body_value.has("stringValue")) {
			log->set_body(body_value["stringValue"]);
		}
	}

	if (p_dict.has("attributes")) {
		log->set_attributes(p_dict["attributes"]);
	}

	if (p_dict.has("traceId")) {
		log->set_trace_id(p_dict["traceId"]);
	}

	if (p_dict.has("spanId")) {
		log->set_span_id(p_dict["spanId"]);
	}

	return log;
}
