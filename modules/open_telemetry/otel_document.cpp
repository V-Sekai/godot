/**************************************************************************/
/*  otel_document.cpp                                                     */
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

#include "otel_document.h"

#include "otel_state.h"
#include "structures/otel_resource.h"
#include "structures/otel_scope.h"
#include "structures/otel_span.h"

#include "core/io/json.h"

void OTelDocument::_bind_methods() {
	// Serialization
	ClassDB::bind_method(D_METHOD("serialize_traces", "state"), &OTelDocument::serialize_traces);
	ClassDB::bind_method(D_METHOD("serialize_metrics", "state"), &OTelDocument::serialize_metrics);
	ClassDB::bind_method(D_METHOD("serialize_logs", "state"), &OTelDocument::serialize_logs);

	// Deserialization
	ClassDB::bind_method(D_METHOD("deserialize_traces", "json"), &OTelDocument::deserialize_traces);
	ClassDB::bind_method(D_METHOD("deserialize_metrics", "json"), &OTelDocument::deserialize_metrics);
	ClassDB::bind_method(D_METHOD("deserialize_logs", "json"), &OTelDocument::deserialize_logs);

	// Build payloads
	ClassDB::bind_method(D_METHOD("build_trace_payload", "resource", "scope", "spans"), &OTelDocument::build_trace_payload);
	ClassDB::bind_method(D_METHOD("build_metric_payload", "resource", "scope", "metrics"), &OTelDocument::build_metric_payload);
	ClassDB::bind_method(D_METHOD("build_log_payload", "resource", "scope", "logs"), &OTelDocument::build_log_payload);
}

OTelDocument::OTelDocument() {
}

// Helper: Convert single attribute to OTLP format
Dictionary OTelDocument::attribute_to_otlp(const String &p_key, const Variant &p_value) {
	Dictionary attr;
	attr["key"] = p_key;

	Dictionary value_dict;
	switch (p_value.get_type()) {
		case Variant::STRING:
			value_dict["stringValue"] = String(p_value);
			break;
		case Variant::INT:
			value_dict["intValue"] = (int64_t)p_value;
			break;
		case Variant::FLOAT:
			value_dict["doubleValue"] = (double)p_value;
			break;
		case Variant::BOOL:
			value_dict["boolValue"] = (bool)p_value;
			break;
		default:
			value_dict["stringValue"] = String(p_value);
			break;
	}

	attr["value"] = value_dict;
	return attr;
}

// Helper: Convert Dictionary of attributes to OTLP array format
Array OTelDocument::attributes_to_otlp(const Dictionary &p_attributes) {
	Array otlp_attributes;
	Array keys = p_attributes.keys();

	for (int i = 0; i < keys.size(); i++) {
		String key = keys[i];
		Variant value = p_attributes[key];
		otlp_attributes.push_back(attribute_to_otlp(key, value));
	}

	return otlp_attributes;
}

// Helper: Convert OTLP attribute array to Dictionary
Dictionary OTelDocument::attributes_from_otlp(const Array &p_otlp_attributes) {
	Dictionary attributes;

	for (int i = 0; i < p_otlp_attributes.size(); i++) {
		Dictionary attr = p_otlp_attributes[i];
		if (attr.has("key") && attr.has("value")) {
			String key = attr["key"];
			Dictionary value_dict = attr["value"];

			Variant value;
			if (value_dict.has("stringValue")) {
				value = value_dict["stringValue"];
			} else if (value_dict.has("intValue")) {
				value = value_dict["intValue"];
			} else if (value_dict.has("doubleValue")) {
				value = value_dict["doubleValue"];
			} else if (value_dict.has("boolValue")) {
				value = value_dict["boolValue"];
			}

			attributes[key] = value;
		}
	}

	return attributes;
}

// Build complete OTLP trace payload
Dictionary OTelDocument::build_trace_payload(Ref<OTelResource> p_resource, Ref<OTelScope> p_scope, const TypedArray<OTelSpan> &p_spans) {
	// Build spans array using OTelSpan::to_otlp_dict()
	Array spans_array;
	for (int i = 0; i < p_spans.size(); i++) {
		Ref<OTelSpan> span = p_spans[i];
		if (!span.is_valid()) {
			continue;
		}
		spans_array.push_back(span->to_otlp_dict());
	}

	// Build scope spans
	Dictionary scope_span;
	scope_span["scope"] = p_scope->to_otlp_dict();
	scope_span["spans"] = spans_array;

	Array scope_spans;
	scope_spans.push_back(scope_span);

	// Build resource spans
	Dictionary resource_span;
	resource_span["resource"] = p_resource->to_otlp_dict();
	resource_span["scopeSpans"] = scope_spans;

	Array resource_spans;
	resource_spans.push_back(resource_span);

	// Build root
	Dictionary root;
	root["resourceSpans"] = resource_spans;

	return root;
}

// Build complete OTLP metric payload
Dictionary OTelDocument::build_metric_payload(Ref<OTelResource> p_resource, Ref<OTelScope> p_scope, const Array &p_metrics) {
	// Build scope metrics
	Dictionary scope_metric;
	scope_metric["scope"] = p_scope->to_otlp_dict();
	scope_metric["metrics"] = p_metrics;

	Array scope_metrics;
	scope_metrics.push_back(scope_metric);

	// Build resource metrics
	Dictionary resource_metric;
	resource_metric["resource"] = p_resource->to_otlp_dict();
	resource_metric["scopeMetrics"] = scope_metrics;

	Array resource_metrics;
	resource_metrics.push_back(resource_metric);

	// Build root
	Dictionary root;
	root["resourceMetrics"] = resource_metrics;

	return root;
}

// Build complete OTLP log payload
Dictionary OTelDocument::build_log_payload(Ref<OTelResource> p_resource, Ref<OTelScope> p_scope, const Array &p_logs) {
	// Build scope logs
	Dictionary scope_log;
	scope_log["scope"] = p_scope->to_otlp_dict();
	scope_log["logRecords"] = p_logs;

	Array scope_logs;
	scope_logs.push_back(scope_log);

	// Build resource logs
	Dictionary resource_log;
	resource_log["resource"] = p_resource->to_otlp_dict();
	resource_log["scopeLogs"] = scope_logs;

	Array resource_logs;
	resource_logs.push_back(resource_log);

	// Build root
	Dictionary root;
	root["resourceLogs"] = resource_logs;

	return root;
}

// Serialize traces to OTLP JSON
String OTelDocument::serialize_traces(Ref<OTelState> p_state) {
	if (p_state.is_null()) {
		return "{}";
	}

	// Access internal Vector directly (friend class)
	Array spans_array;
	for (int i = 0; i < p_state->spans.size(); i++) {
		if (!p_state->spans[i].is_valid()) {
			continue;
		}
		spans_array.push_back(p_state->spans[i]->to_otlp_dict());
	}

	// Build scope spans
	Dictionary scope_span;
	scope_span["scope"] = p_state->scope->to_otlp_dict();
	scope_span["spans"] = spans_array;

	Array scope_spans;
	scope_spans.push_back(scope_span);

	// Build resource spans
	Dictionary resource_span;
	resource_span["resource"] = p_state->resource->to_otlp_dict();
	resource_span["scopeSpans"] = scope_spans;

	Array resource_spans;
	resource_spans.push_back(resource_span);

	// Build root
	Dictionary root;
	root["resourceSpans"] = resource_spans;

	return JSON::stringify(root);
}

// Serialize metrics to OTLP JSON
String OTelDocument::serialize_metrics(Ref<OTelState> p_state) {
	if (p_state.is_null()) {
		return "{}";
	}

	// Access internal Vector directly (friend class)
	Array metrics_array;
	for (int i = 0; i < p_state->metrics.size(); i++) {
		if (!p_state->metrics[i].is_valid()) {
			continue;
		}
		metrics_array.push_back(p_state->metrics[i]->to_otlp_dict());
	}

	// Build scope metrics
	Dictionary scope_metric;
	scope_metric["scope"] = p_state->scope->to_otlp_dict();
	scope_metric["metrics"] = metrics_array;

	Array scope_metrics;
	scope_metrics.push_back(scope_metric);

	// Build resource metrics
	Dictionary resource_metric;
	resource_metric["resource"] = p_state->resource->to_otlp_dict();
	resource_metric["scopeMetrics"] = scope_metrics;

	Array resource_metrics;
	resource_metrics.push_back(resource_metric);

	// Build root
	Dictionary root;
	root["resourceMetrics"] = resource_metrics;

	return JSON::stringify(root);
}

// Serialize logs to OTLP JSON
String OTelDocument::serialize_logs(Ref<OTelState> p_state) {
	if (p_state.is_null()) {
		return "{}";
	}

	// Access internal Vector directly (friend class)
	Array logs_array;
	for (int i = 0; i < p_state->logs.size(); i++) {
		if (!p_state->logs[i].is_valid()) {
			continue;
		}
		logs_array.push_back(p_state->logs[i]->to_otlp_dict());
	}

	// Build scope logs
	Dictionary scope_log;
	scope_log["scope"] = p_state->scope->to_otlp_dict();
	scope_log["logRecords"] = logs_array;

	Array scope_logs;
	scope_logs.push_back(scope_log);

	// Build resource logs
	Dictionary resource_log;
	resource_log["resource"] = p_state->resource->to_otlp_dict();
	resource_log["scopeLogs"] = scope_logs;

	Array resource_logs;
	resource_logs.push_back(resource_log);

	// Build root
	Dictionary root;
	root["resourceLogs"] = resource_logs;

	return JSON::stringify(root);
}

// Deserialize traces from OTLP JSON (for reflector)
Ref<OTelState> OTelDocument::deserialize_traces(const String &p_json) {
	Ref<OTelState> state;
	state.instantiate();

	JSON json;
	Error err = json.parse(p_json);
	if (err != OK) {
		return state;
	}

	Dictionary root = json.get_data();
	if (!root.has("resourceSpans")) {
		return state;
	}

	Array resource_spans = root["resourceSpans"];
	if (resource_spans.size() == 0) {
		return state;
	}

	Dictionary resource_span = resource_spans[0];

	// Parse resource
	if (resource_span.has("resource")) {
		state->set_resource(OTelResource::from_otlp_dict(resource_span["resource"]));
	}

	// Parse scope spans
	if (!resource_span.has("scopeSpans")) {
		return state;
	}

	Array scope_spans = resource_span["scopeSpans"];
	if (scope_spans.size() == 0) {
		return state;
	}

	Dictionary scope_span = scope_spans[0];

	// Parse scope
	if (scope_span.has("scope")) {
		state->set_scope(OTelScope::from_otlp_dict(scope_span["scope"]));
	}

	// Parse spans
	if (!scope_span.has("spans")) {
		return state;
	}

	Array spans = scope_span["spans"];
	TypedArray<OTelSpan> otel_spans;

	for (int i = 0; i < spans.size(); i++) {
		Dictionary span_dict = spans[i];
		otel_spans.push_back(OTelSpan::from_otlp_dict(span_dict));
	}

	state->set_spans(otel_spans);

	return state;
}

// Deserialize metrics from OTLP JSON (for reflector)
Ref<OTelState> OTelDocument::deserialize_metrics(const String &p_json) {
	Ref<OTelState> state;
	state.instantiate();

	JSON json;
	Error err = json.parse(p_json);
	if (err != OK) {
		return state;
	}

	Dictionary root = json.get_data();
	if (!root.has("resourceMetrics")) {
		return state;
	}

	Array resource_metrics = root["resourceMetrics"];
	if (resource_metrics.size() == 0) {
		return state;
	}

	Dictionary resource_metric = resource_metrics[0];

	// Parse resource
	if (resource_metric.has("resource")) {
		state->set_resource(OTelResource::from_otlp_dict(resource_metric["resource"]));
	}

	// Parse scope metrics
	if (!resource_metric.has("scopeMetrics")) {
		return state;
	}

	Array scope_metrics = resource_metric["scopeMetrics"];
	if (scope_metrics.size() == 0) {
		return state;
	}

	Dictionary scope_metric = scope_metrics[0];

	// Parse scope
	if (scope_metric.has("scope")) {
		state->set_scope(OTelScope::from_otlp_dict(scope_metric["scope"]));
	}

	// Parse metrics
	if (scope_metric.has("metrics")) {
		state->set_metrics(scope_metric["metrics"]);
	}

	return state;
}

// Deserialize logs from OTLP JSON (for reflector)
Ref<OTelState> OTelDocument::deserialize_logs(const String &p_json) {
	Ref<OTelState> state;
	state.instantiate();

	JSON json;
	Error err = json.parse(p_json);
	if (err != OK) {
		return state;
	}

	Dictionary root = json.get_data();
	if (!root.has("resourceLogs")) {
		return state;
	}

	Array resource_logs = root["resourceLogs"];
	if (resource_logs.size() == 0) {
		return state;
	}

	Dictionary resource_log = resource_logs[0];

	// Parse resource
	if (resource_log.has("resource")) {
		state->set_resource(OTelResource::from_otlp_dict(resource_log["resource"]));
	}

	// Parse scope logs
	if (!resource_log.has("scopeLogs")) {
		return state;
	}

	Array scope_logs = resource_log["scopeLogs"];
	if (scope_logs.size() == 0) {
		return state;
	}

	Dictionary scope_log = scope_logs[0];

	// Parse scope
	if (scope_log.has("scope")) {
		state->set_scope(OTelScope::from_otlp_dict(scope_log["scope"]));
	}

	// Parse logs
	if (scope_log.has("logRecords")) {
		state->set_logs(scope_log["logRecords"]);
	}

	return state;
}
