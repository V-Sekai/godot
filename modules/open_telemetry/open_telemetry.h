/**************************************************************************/
/*  open_telemetry.h                                                      */
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

#include "core/crypto/crypto.h"
#include "core/io/http_client.h"
#include "core/io/json.h"
#include "core/object/class_db.h"
#include "core/object/method_bind.h"
#include "core/object/ref_counted.h"
#include "core/os/time.h"
#include "core/templates/cowdata.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

// New structure classes
#include "otel_document.h"
#include "otel_state.h"
#include "structures/otel_span.h"

// Deprecated enums - kept for backward compatibility, map to OTelSpan enums
enum SpanKind {
	SPAN_KIND_INTERNAL = 0,
	SPAN_KIND_SERVER = 1,
	SPAN_KIND_CLIENT = 2,
	SPAN_KIND_PRODUCER = 3,
	SPAN_KIND_CONSUMER = 4,
};

enum StatusCode {
	STATUS_UNSET = 0,
	STATUS_OK = 1,
	STATUS_ERROR = 2,
};

class OpenTelemetryTracer : public RefCounted {
	GDCLASS(OpenTelemetryTracer, RefCounted);

private:
	String name;
	String version;
	String schema_url;
	Dictionary attributes;

public:
	OpenTelemetryTracer(String p_name, String p_version = "", String p_schema_url = "", Dictionary p_attributes = Dictionary());

	String get_name() const { return name; }
	String get_version() const { return version; }
	String get_schema_url() const { return schema_url; }
	Dictionary get_attributes() const { return attributes; }
	bool enabled() const { return true; }

protected:
	static void _bind_methods();
};

class OpenTelemetryTracerProvider : public RefCounted {
	GDCLASS(OpenTelemetryTracerProvider, RefCounted);

private:
	Dictionary tracers;
	Dictionary resource_attributes;

public:
	OpenTelemetryTracerProvider(Dictionary p_resource_attributes = Dictionary());

	Ref<OpenTelemetryTracer> get_tracer(String p_name, String p_version = "", String p_schema_url = "", Dictionary p_attributes = Dictionary());
	Dictionary get_resource_attributes() const { return resource_attributes; }
	void set_resource_attributes(Dictionary p_attributes) { resource_attributes = p_attributes; }

protected:
	static void _bind_methods();
};

class OpenTelemetry : public RefCounted {
	GDCLASS(OpenTelemetry, RefCounted);

private:
	// New structure-based state
	Ref<OTelState> state;
	Ref<OTelDocument> document;

	// Active spans tracking (maps UUID -> OTelSpan)
	Dictionary active_spans;

	// Configuration
	String hostname;
	Dictionary resource_attributes;
	Dictionary headers;
	String trace_id;
	String tracer_name;
	int flush_interval_ms;
	int batch_size;
	uint64_t last_flush_time;

	// Multi-sink support
	Dictionary sinks;

	// HTTP client for OTLP export
	Ref<HTTPClient> http_client;

protected:
	static void _bind_methods();

public:
	OpenTelemetry();
	~OpenTelemetry();

	String init_tracer_provider(String p_name, String p_host, Dictionary p_attributes, String p_version = "", String p_schema_url = "");
	String set_headers(Dictionary p_headers);
	String start_span(String p_name, SpanKind p_kind = SPAN_KIND_INTERNAL, Array p_links = Array(), Dictionary p_attributes = Dictionary());
	String start_span_with_parent(String p_name, String p_parent_span_uuid, SpanKind p_kind = SPAN_KIND_INTERNAL, Array p_links = Array(), Dictionary p_attributes = Dictionary());
	String generate_uuid_v7();
	void update_name(String p_span_uuid, String p_name);
	void add_event(String p_span_uuid, String p_event_name, Dictionary p_attributes = Dictionary(), uint64_t p_timestamp = 0);
	void record_exception(String p_span_uuid, String p_message, Dictionary p_attributes = Dictionary());
	void set_attributes(String p_span_uuid, Dictionary p_attributes);
	void set_status(String p_span_uuid, int p_status_code, String p_description = "");
	void record_error(String p_span_uuid, String p_error);
	void end_span(String p_span_uuid);
	void set_flush_interval(int p_interval_ms);
	void set_batch_size(int p_size);
	void record_metric(String p_name, float p_value, String p_unit, int p_metric_type, Dictionary p_attributes);
	void log_message(String p_level, String p_message, Dictionary p_attributes);
	void flush_all();
	String shutdown();

	// Metrics API
	String create_counter(String p_name, String p_unit = "", String p_description = "");
	String create_gauge(String p_name, String p_unit = "", String p_description = "");
	String create_histogram(String p_name, String p_unit = "", String p_description = "");
	void increment_counter(String p_counter_id, float p_value = 1.0, Dictionary p_attributes = Dictionary());
	void set_gauge(String p_gauge_id, float p_value, Dictionary p_attributes = Dictionary());
	void record_histogram(String p_histogram_id, float p_value, Dictionary p_attributes = Dictionary());

	// Multi-sink API
	String add_sink(String p_sink_name, String p_hostname, Dictionary p_headers = Dictionary());
	String remove_sink(String p_sink_name);
	String set_sink_enabled(String p_sink_name, bool p_enabled);
	Dictionary get_sink(String p_sink_name);
	Array list_sinks();

	// Direct access to new classes (for advanced usage)
	Ref<OTelState> get_state() const { return state; }
	Ref<OTelDocument> get_document() const { return document; }

private:
	// OTLP ID generation helpers
	String generate_trace_id();
	String generate_span_id();

	// HTTP/OTLP export helpers
	Error _send_otlp_request(const String &p_endpoint, const String &p_json_body);
	Error _send_to_sink(const String &p_sink_hostname, const Dictionary &p_sink_headers, const String &p_endpoint, const String &p_json_body);

	String start_span_with_id(String p_name, String p_span_id);
	String start_span_with_parent_id(String p_name, String p_parent_span_uuid, String p_span_id);

	void CheckAndFlush();
	void FlushAllBufferedData();
};

VARIANT_ENUM_CAST(StatusCode);
VARIANT_ENUM_CAST(SpanKind);
