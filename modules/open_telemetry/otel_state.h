/**************************************************************************/
/*  otel_state.h                                                         */
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

#ifndef OTEL_STATE_H
#define OTEL_STATE_H

#include "core/io/resource.h"
#include "core/variant/typed_array.h"
#include "structures/otel_log.h"
#include "structures/otel_metric.h"
#include "structures/otel_resource.h"
#include "structures/otel_scope.h"
#include "structures/otel_span.h"

// OTelState manages runtime state of telemetry data
// Similar to GLTFState - holds active spans, metrics, logs, and configuration
class OTelState : public Resource {
	GDCLASS(OTelState, Resource);
	friend class OTelDocument;

private:
	// Resource configuration (service.name, etc.)
	Ref<OTelResource> resource;

	// Instrumentation scope
	Ref<OTelScope> scope;

	// Telemetry data - C++ Vector internally, convert to Array at API boundary
	Vector<Ref<OTelSpan>> spans;
	Vector<Ref<OTelMetric>> metrics;
	Vector<Ref<OTelLog>> logs;

protected:
	static void _bind_methods();

public:
	OTelState();

	// Resource management
	Ref<OTelResource> get_resource() const;
	void set_resource(Ref<OTelResource> p_resource);

	// Scope management
	Ref<OTelScope> get_scope() const;
	void set_scope(Ref<OTelScope> p_scope);

	// Span management
	TypedArray<OTelSpan> get_spans() const;
	void set_spans(const TypedArray<OTelSpan> &p_spans);
	void add_span(Ref<OTelSpan> p_span);
	void clear_spans();

	// Metric management
	TypedArray<OTelMetric> get_metrics() const;
	void set_metrics(const TypedArray<OTelMetric> &p_metrics);
	void add_metric(Ref<OTelMetric> p_metric);
	void clear_metrics();

	// Log management
	TypedArray<OTelLog> get_logs() const;
	void set_logs(const TypedArray<OTelLog> &p_logs);
	void add_log(Ref<OTelLog> p_log);
	void clear_logs();

	// Utility
	void clear_all();
};

#endif // OTEL_STATE_H
