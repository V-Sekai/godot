/**************************************************************************/
/*  otel_state.cpp                                                        */
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

#include "otel_state.h"

#include "structures/otel_resource.h"
#include "structures/otel_scope.h"
#include "structures/otel_span.h"

void OpenTelemetryState::_bind_methods() {
	// Resource management
	ClassDB::bind_method(D_METHOD("get_resource"), &OpenTelemetryState::get_resource);
	ClassDB::bind_method(D_METHOD("set_resource", "resource"), &OpenTelemetryState::set_resource);

	// Scope management
	ClassDB::bind_method(D_METHOD("get_scope"), &OpenTelemetryState::get_scope);
	ClassDB::bind_method(D_METHOD("set_scope", "scope"), &OpenTelemetryState::set_scope);

	// Span management
	ClassDB::bind_method(D_METHOD("get_spans"), &OpenTelemetryState::get_spans);
	ClassDB::bind_method(D_METHOD("set_spans", "spans"), &OpenTelemetryState::set_spans);
	ClassDB::bind_method(D_METHOD("add_span", "span"), &OpenTelemetryState::add_span);
	ClassDB::bind_method(D_METHOD("clear_spans"), &OpenTelemetryState::clear_spans);

	// Metric management
	ClassDB::bind_method(D_METHOD("get_metrics"), &OpenTelemetryState::get_metrics);
	ClassDB::bind_method(D_METHOD("set_metrics", "metrics"), &OpenTelemetryState::set_metrics);
	ClassDB::bind_method(D_METHOD("add_metric", "metric"), &OpenTelemetryState::add_metric);
	ClassDB::bind_method(D_METHOD("clear_metrics"), &OpenTelemetryState::clear_metrics);

	// Log management
	ClassDB::bind_method(D_METHOD("get_logs"), &OpenTelemetryState::get_logs);
	ClassDB::bind_method(D_METHOD("set_logs", "logs"), &OpenTelemetryState::set_logs);
	ClassDB::bind_method(D_METHOD("add_log", "log"), &OpenTelemetryState::add_log);
	ClassDB::bind_method(D_METHOD("clear_logs"), &OpenTelemetryState::clear_logs);

	// Utility
	ClassDB::bind_method(D_METHOD("clear_all"), &OpenTelemetryState::clear_all);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "OpenTelemetryResource"), "set_resource", "get_resource");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "scope", PROPERTY_HINT_RESOURCE_TYPE, "OpenTelemetryScope"), "set_scope", "get_scope");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "spans", PROPERTY_HINT_ARRAY_TYPE, "OpenTelemetrySpan"), "set_spans", "get_spans");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "metrics"), "set_metrics", "get_metrics");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "logs"), "set_logs", "get_logs");
}

OpenTelemetryState::OpenTelemetryState() {
	resource.instantiate();
	scope.instantiate();
}

// Resource management
Ref<OpenTelemetryResource> OpenTelemetryState::get_resource() const {
	return resource;
}

void OpenTelemetryState::set_resource(Ref<OpenTelemetryResource> p_resource) {
	resource = p_resource;
}

// Scope management
Ref<OpenTelemetryScope> OpenTelemetryState::get_scope() const {
	return scope;
}

void OpenTelemetryState::set_scope(Ref<OpenTelemetryScope> p_scope) {
	scope = p_scope;
}

// Span management
TypedArray<OpenTelemetrySpan> OpenTelemetryState::get_spans() const {
	// Convert Vector to TypedArray at API boundary
	TypedArray<OpenTelemetrySpan> result;
	for (int i = 0; i < spans.size(); i++) {
		result.push_back(spans[i]);
	}
	return result;
}

void OpenTelemetryState::set_spans(const TypedArray<OpenTelemetrySpan> &p_spans) {
	// Convert TypedArray to Vector at API boundary
	spans.clear();
	for (int i = 0; i < p_spans.size(); i++) {
		spans.push_back(p_spans[i]);
	}
}

void OpenTelemetryState::add_span(Ref<OpenTelemetrySpan> p_span) {
	if (p_span.is_valid()) {
		spans.push_back(p_span);
	}
}

void OpenTelemetryState::clear_spans() {
	spans.clear();
}

// Metric management
TypedArray<OpenTelemetryMetric> OpenTelemetryState::get_metrics() const {
	// Convert Vector to TypedArray at API boundary
	TypedArray<OpenTelemetryMetric> result;
	for (int i = 0; i < metrics.size(); i++) {
		result.push_back(metrics[i]);
	}
	return result;
}

void OpenTelemetryState::set_metrics(const TypedArray<OpenTelemetryMetric> &p_metrics) {
	// Convert TypedArray to Vector at API boundary
	metrics.clear();
	for (int i = 0; i < p_metrics.size(); i++) {
		metrics.push_back(p_metrics[i]);
	}
}

void OpenTelemetryState::add_metric(Ref<OpenTelemetryMetric> p_metric) {
	if (p_metric.is_valid()) {
		metrics.push_back(p_metric);
	}
}

void OpenTelemetryState::clear_metrics() {
	metrics.clear();
}

// Log management
TypedArray<OpenTelemetryLog> OpenTelemetryState::get_logs() const {
	// Convert Vector to TypedArray at API boundary
	TypedArray<OpenTelemetryLog> result;
	for (int i = 0; i < logs.size(); i++) {
		result.push_back(logs[i]);
	}
	return result;
}

void OpenTelemetryState::set_logs(const TypedArray<OpenTelemetryLog> &p_logs) {
	// Convert TypedArray to Vector at API boundary
	logs.clear();
	for (int i = 0; i < p_logs.size(); i++) {
		logs.push_back(p_logs[i]);
	}
}

void OpenTelemetryState::add_log(Ref<OpenTelemetryLog> p_log) {
	if (p_log.is_valid()) {
		logs.push_back(p_log);
	}
}

void OpenTelemetryState::clear_logs() {
	logs.clear();
}

// Utility
void OpenTelemetryState::clear_all() {
	clear_spans();
	clear_metrics();
	clear_logs();
}
