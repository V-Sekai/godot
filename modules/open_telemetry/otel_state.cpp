/**************************************************************************/
/*  otel_state.cpp                                                       */
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

void OTelState::_bind_methods() {
	// Resource management
	ClassDB::bind_method(D_METHOD("get_resource"), &OTelState::get_resource);
	ClassDB::bind_method(D_METHOD("set_resource", "resource"), &OTelState::set_resource);

	// Scope management
	ClassDB::bind_method(D_METHOD("get_scope"), &OTelState::get_scope);
	ClassDB::bind_method(D_METHOD("set_scope", "scope"), &OTelState::set_scope);

	// Span management
	ClassDB::bind_method(D_METHOD("get_spans"), &OTelState::get_spans);
	ClassDB::bind_method(D_METHOD("set_spans", "spans"), &OTelState::set_spans);
	ClassDB::bind_method(D_METHOD("add_span", "span"), &OTelState::add_span);
	ClassDB::bind_method(D_METHOD("clear_spans"), &OTelState::clear_spans);

	// Metric management
	ClassDB::bind_method(D_METHOD("get_metrics"), &OTelState::get_metrics);
	ClassDB::bind_method(D_METHOD("set_metrics", "metrics"), &OTelState::set_metrics);
	ClassDB::bind_method(D_METHOD("add_metric", "metric"), &OTelState::add_metric);
	ClassDB::bind_method(D_METHOD("clear_metrics"), &OTelState::clear_metrics);

	// Log management
	ClassDB::bind_method(D_METHOD("get_logs"), &OTelState::get_logs);
	ClassDB::bind_method(D_METHOD("set_logs", "logs"), &OTelState::set_logs);
	ClassDB::bind_method(D_METHOD("add_log", "log"), &OTelState::add_log);
	ClassDB::bind_method(D_METHOD("clear_logs"), &OTelState::clear_logs);

	// Utility
	ClassDB::bind_method(D_METHOD("clear_all"), &OTelState::clear_all);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "OTelResource"), "set_resource", "get_resource");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "scope", PROPERTY_HINT_RESOURCE_TYPE, "OTelScope"), "set_scope", "get_scope");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "spans", PROPERTY_HINT_ARRAY_TYPE, "OTelSpan"), "set_spans", "get_spans");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "metrics"), "set_metrics", "get_metrics");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "logs"), "set_logs", "get_logs");
}

OTelState::OTelState() {
	resource.instantiate();
	scope.instantiate();
}

// Resource management
Ref<OTelResource> OTelState::get_resource() const {
	return resource;
}

void OTelState::set_resource(Ref<OTelResource> p_resource) {
	resource = p_resource;
}

// Scope management
Ref<OTelScope> OTelState::get_scope() const {
	return scope;
}

void OTelState::set_scope(Ref<OTelScope> p_scope) {
	scope = p_scope;
}

// Span management
TypedArray<OTelSpan> OTelState::get_spans() const {
	// Convert Vector to TypedArray at API boundary
	TypedArray<OTelSpan> result;
	for (int i = 0; i < spans.size(); i++) {
		result.push_back(spans[i]);
	}
	return result;
}

void OTelState::set_spans(const TypedArray<OTelSpan> &p_spans) {
	// Convert TypedArray to Vector at API boundary
	spans.clear();
	for (int i = 0; i < p_spans.size(); i++) {
		spans.push_back(p_spans[i]);
	}
}

void OTelState::add_span(Ref<OTelSpan> p_span) {
	if (p_span.is_valid()) {
		spans.push_back(p_span);
	}
}

void OTelState::clear_spans() {
	spans.clear();
}

// Metric management
TypedArray<OTelMetric> OTelState::get_metrics() const {
	// Convert Vector to TypedArray at API boundary
	TypedArray<OTelMetric> result;
	for (int i = 0; i < metrics.size(); i++) {
		result.push_back(metrics[i]);
	}
	return result;
}

void OTelState::set_metrics(const TypedArray<OTelMetric> &p_metrics) {
	// Convert TypedArray to Vector at API boundary
	metrics.clear();
	for (int i = 0; i < p_metrics.size(); i++) {
		metrics.push_back(p_metrics[i]);
	}
}

void OTelState::add_metric(Ref<OTelMetric> p_metric) {
	if (p_metric.is_valid()) {
		metrics.push_back(p_metric);
	}
}

void OTelState::clear_metrics() {
	metrics.clear();
}

// Log management
TypedArray<OTelLog> OTelState::get_logs() const {
	// Convert Vector to TypedArray at API boundary
	TypedArray<OTelLog> result;
	for (int i = 0; i < logs.size(); i++) {
		result.push_back(logs[i]);
	}
	return result;
}

void OTelState::set_logs(const TypedArray<OTelLog> &p_logs) {
	// Convert TypedArray to Vector at API boundary
	logs.clear();
	for (int i = 0; i < p_logs.size(); i++) {
		logs.push_back(p_logs[i]);
	}
}

void OTelState::add_log(Ref<OTelLog> p_log) {
	if (p_log.is_valid()) {
		logs.push_back(p_log);
	}
}

void OTelState::clear_logs() {
	logs.clear();
}

// Utility
void OTelState::clear_all() {
	clear_spans();
	clear_metrics();
	clear_logs();
}
