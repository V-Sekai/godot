/**************************************************************************/
/*  open_telemetry.cpp                                                    */
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

#include "open_telemetry.h"

void OpenTelemetry::_bind_methods() {
	ClassDB::bind_method(D_METHOD("init_tracer_provider", "name", "host", "attributes"), &OpenTelemetry::init_tracer_provider);
	ClassDB::bind_method(D_METHOD("start_span", "name"), &OpenTelemetry::start_span);
	ClassDB::bind_method(D_METHOD("start_span_with_parent", "name", "parent_span_uuid"), &OpenTelemetry::start_span_with_parent);
	ClassDB::bind_method(D_METHOD("add_event", "span_uuid", "event_name"), &OpenTelemetry::add_event);
	ClassDB::bind_method(D_METHOD("set_attributes", "span_uuid", "attributes"), &OpenTelemetry::set_attributes);
	ClassDB::bind_method(D_METHOD("record_error", "span_uuid", "err"), &OpenTelemetry::record_error);
	ClassDB::bind_method(D_METHOD("end_span", "span_uuid"), &OpenTelemetry::end_span);
	ClassDB::bind_method(D_METHOD("shutdown"), &OpenTelemetry::shutdown);
}

String OpenTelemetry::init_tracer_provider(String p_name, String p_host, Dictionary p_attributes) {
	CharString cs = p_name.utf8();
	char *cstr = cs.ptrw();
	CharString c_host = p_host.utf8();
	char *cstr_host = c_host.ptrw();
	String json_attributes = JSON::stringify(p_attributes, "", true, true);
	CharString c_json_attributes = json_attributes.utf8();
	char *cstr_json_attributes = c_json_attributes.ptrw();
	const char *result = InitTracerProvider(cstr, cstr_host, cstr_json_attributes);
	return String(result);
}

String OpenTelemetry::start_span(String p_name) {
	CharString c_span_name = p_name.utf8();
	char *cstr_span_name = c_span_name.ptrw();
	char *result = StartSpan(cstr_span_name);
	return String(result);
}

String OpenTelemetry::start_span_with_parent(String p_name, String p_parent_span_uuid) {
	CharString c_with_parent_name = p_name.utf8();
	char *cstr_with_parent_name = c_with_parent_name.ptrw();
	CharString c_parent_id = p_parent_span_uuid.utf8();
	char *cstr_parent_id = c_parent_id.ptrw();
	char *result = StartSpanWithParent(cstr_with_parent_name, cstr_parent_id);
	return String(result);
}

void OpenTelemetry::add_event(String p_span_uuid, String p_event_name) {
	CharString c_event_id = p_span_uuid.utf8();
	char *cstr_event_id = c_event_id.ptrw();
	CharString c_event_name = p_event_name.utf8();
	char *cstr_event_name = c_event_name.ptrw();
	AddEvent(cstr_event_id, cstr_event_name);
}

void OpenTelemetry::set_attributes(String p_span_uuid, Dictionary p_attributes) {
	CharString c_attribute_id = p_span_uuid.utf8();
	char *cstr_attribute_id = c_attribute_id.ptrw();
	String json_attributes = JSON::stringify(p_attributes, "", true, true);
	CharString c_json_attributes = json_attributes.utf8();
	char *cstr_json_attributes = c_json_attributes.ptrw();
	SetAttributes(cstr_attribute_id, cstr_json_attributes);
}

void OpenTelemetry::record_error(String p_span_uuid, String p_error) {
	CharString c_error_id = p_span_uuid.utf8();
	char *cstr_error_id = c_error_id.ptrw();
	CharString c_error = p_error.utf8();
	char *cstr_error = c_error.ptrw();
	RecordError(cstr_error_id, cstr_error);
}

void OpenTelemetry::end_span(String p_span_uuid) {
	CharString c_span_id = p_span_uuid.utf8();
	char *cstr_span_id = c_span_id.ptrw();
	EndSpan(cstr_span_id);
}

String OpenTelemetry::shutdown() {
	char *shutdown_result = Shutdown();
	return String(shutdown_result);
}
