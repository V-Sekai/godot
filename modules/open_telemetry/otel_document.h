/**************************************************************************/
/*  otel_document.h                                                       */
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

class OpenTelemetryState;
class OpenTelemetrySpan;
class OpenTelemetryResource;
class OpenTelemetryScope;

// OpenTelemetryDocument handles serialization/deserialization of OTLP data
// Similar to GLTFDocument - central conversion between internal structures and OTLP JSON
class OpenTelemetryDocument : public Resource {
	GDCLASS(OpenTelemetryDocument, Resource);

protected:
	static void _bind_methods();

public:
	OpenTelemetryDocument();

	// Serialization: Internal structures -> OTLP JSON
	String serialize_traces(Ref<OpenTelemetryState> p_state);
	String serialize_metrics(Ref<OpenTelemetryState> p_state);
	String serialize_logs(Ref<OpenTelemetryState> p_state);

	// Deserialization: OTLP JSON -> Internal structures (for reflector)
	Ref<OpenTelemetryState> deserialize_traces(const String &p_json);
	Ref<OpenTelemetryState> deserialize_metrics(const String &p_json);
	Ref<OpenTelemetryState> deserialize_logs(const String &p_json);

	// Build complete OTLP payload
	Dictionary build_trace_payload(Ref<OpenTelemetryResource> p_resource, Ref<OpenTelemetryScope> p_scope, const TypedArray<OpenTelemetrySpan> &p_spans);
	Dictionary build_metric_payload(Ref<OpenTelemetryResource> p_resource, Ref<OpenTelemetryScope> p_scope, const Array &p_metrics);
	Dictionary build_log_payload(Ref<OpenTelemetryResource> p_resource, Ref<OpenTelemetryScope> p_scope, const Array &p_logs);

	// Helper methods
	static Dictionary attribute_to_otlp(const String &p_key, const Variant &p_value);
	static Array attributes_to_otlp(const Dictionary &p_attributes);
	static Dictionary attributes_from_otlp(const Array &p_otlp_attributes);
};
