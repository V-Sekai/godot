/**************************************************************************/
/*  otel_state.h                                                          */
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

#pragma once

#include "core/io/resource.h"
#include "core/variant/typed_array.h"
#include "structures/otel_log.h"
#include "structures/otel_metric.h"
#include "structures/otel_resource.h"
#include "structures/otel_scope.h"
#include "structures/otel_span.h"

// OpenTelemetryState manages runtime state of telemetry data
// Similar to GLTFState - holds active spans, metrics, logs, and configuration
class OpenTelemetryState : public Resource {
	GDCLASS(OpenTelemetryState, Resource);
	friend class OpenTelemetryDocument;

private:
	// Resource configuration (service.name, etc.)
	Ref<OpenTelemetryResource> resource;

	// Instrumentation scope
	Ref<OpenTelemetryScope> scope;

	// Telemetry data - C++ Vector internally, convert to Array at API boundary
	Vector<Ref<OpenTelemetrySpan>> spans;
	Vector<Ref<OpenTelemetryMetric>> metrics;
	Vector<Ref<OpenTelemetryLog>> logs;

protected:
	static void _bind_methods();

public:
	OpenTelemetryState();

	// Resource management
	Ref<OpenTelemetryResource> get_resource() const;
	void set_resource(Ref<OpenTelemetryResource> p_resource);

	// Scope management
	Ref<OpenTelemetryScope> get_scope() const;
	void set_scope(Ref<OpenTelemetryScope> p_scope);

	// Span management
	TypedArray<OpenTelemetrySpan> get_spans() const;
	void set_spans(const TypedArray<OpenTelemetrySpan> &p_spans);
	void add_span(Ref<OpenTelemetrySpan> p_span);
	void clear_spans();

	// Metric management
	TypedArray<OpenTelemetryMetric> get_metrics() const;
	void set_metrics(const TypedArray<OpenTelemetryMetric> &p_metrics);
	void add_metric(Ref<OpenTelemetryMetric> p_metric);
	void clear_metrics();

	// Log management
	TypedArray<OpenTelemetryLog> get_logs() const;
	void set_logs(const TypedArray<OpenTelemetryLog> &p_logs);
	void add_log(Ref<OpenTelemetryLog> p_log);
	void clear_logs();

	// Utility
	void clear_all();
};
