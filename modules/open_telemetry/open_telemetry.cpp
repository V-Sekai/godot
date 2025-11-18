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

#include "core/string/string_builder.h"
#include "structures/otel_log.h"
#include "structures/otel_metric.h"
#include "structures/otel_resource.h"
#include "structures/otel_scope.h"

// OpenTelemetryTracer implementation

OpenTelemetryTracer::OpenTelemetryTracer() {
	name = "";
	version = "";
	schema_url = "";
	attributes = Dictionary();
}

OpenTelemetryTracer::OpenTelemetryTracer(String p_name, String p_version, String p_schema_url, Dictionary p_attributes) {
	name = p_name;
	version = p_version;
	schema_url = p_schema_url;
	attributes = p_attributes;
}

void OpenTelemetryTracer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &OpenTelemetryTracer::get_name);
	ClassDB::bind_method(D_METHOD("get_version"), &OpenTelemetryTracer::get_version);
	ClassDB::bind_method(D_METHOD("get_schema_url"), &OpenTelemetryTracer::get_schema_url);
	ClassDB::bind_method(D_METHOD("get_attributes"), &OpenTelemetryTracer::get_attributes);
	ClassDB::bind_method(D_METHOD("enabled"), &OpenTelemetryTracer::enabled);
}

// OpenTelemetryTracerProvider implementation

OpenTelemetryTracerProvider::OpenTelemetryTracerProvider(Dictionary p_resource_attributes) {
	resource_attributes = p_resource_attributes;
}

Ref<OpenTelemetryTracer> OpenTelemetryTracerProvider::get_tracer(String p_name, String p_version, String p_schema_url, Dictionary p_attributes) {
	String tracer_key = p_name + ":" + p_version;

	if (tracers.has(tracer_key)) {
		return tracers[tracer_key];
	}

	Ref<OpenTelemetryTracer> tracer = memnew(OpenTelemetryTracer(p_name, p_version, p_schema_url, p_attributes));
	tracers[tracer_key] = tracer;
	return tracer;
}

void OpenTelemetryTracerProvider::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_tracer", "name", "version", "schema_url", "attributes"), &OpenTelemetryTracerProvider::get_tracer, DEFVAL(""), DEFVAL(""), DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("get_resource_attributes"), &OpenTelemetryTracerProvider::get_resource_attributes);
	ClassDB::bind_method(D_METHOD("set_resource_attributes", "attributes"), &OpenTelemetryTracerProvider::set_resource_attributes);
}

// OpenTelemetry implementation

OpenTelemetry::OpenTelemetry() {
	// Initialize new structure classes
	state.instantiate();
	document.instantiate();

	// Set default resource and scope
	state->get_resource()->set_service_name("godot_service");
	state->get_scope()->set_name("godot_tracer");

	// Initialize configuration with SigNoz defaults
	hostname = "https://ingest.us.signoz.cloud";
	headers["Authorization"] = "Bearer CLSbUJpdew0gPq2QLZNvPlwSCwnf16hvPE2Z";
	trace_id = "";
	tracer_name = "";
	flush_interval_ms = 10000;
	batch_size = 100;
	last_flush_time = 0;
}

OpenTelemetry::~OpenTelemetry() {
}

void OpenTelemetry::_bind_methods() {
	ClassDB::bind_method(D_METHOD("init_tracer_provider", "name", "host", "attributes", "version", "schema_url"), &OpenTelemetry::init_tracer_provider, DEFVAL(Dictionary()), DEFVAL(""), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_headers", "headers"), &OpenTelemetry::set_headers);
	ClassDB::bind_method(D_METHOD("start_span", "name", "kind", "links", "attributes"), &OpenTelemetry::start_span, DEFVAL(SPAN_KIND_INTERNAL), DEFVAL(Array()), DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("start_span_with_parent", "name", "parent_span_uuid", "kind", "links", "attributes"), &OpenTelemetry::start_span_with_parent, DEFVAL(SPAN_KIND_INTERNAL), DEFVAL(Array()), DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("generate_uuid_v7"), &OpenTelemetry::generate_uuid_v7);
	ClassDB::bind_method(D_METHOD("update_name", "span_uuid", "name"), &OpenTelemetry::update_name);
	ClassDB::bind_method(D_METHOD("add_event", "span_uuid", "event_name", "attributes", "timestamp"), &OpenTelemetry::add_event, DEFVAL(Dictionary()), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("record_exception", "span_uuid", "message", "attributes"), &OpenTelemetry::record_exception, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("set_attributes", "span_uuid", "attributes"), &OpenTelemetry::set_attributes);
	ClassDB::bind_method(D_METHOD("set_status", "span_uuid", "status_code", "description"), &OpenTelemetry::set_status, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("record_error", "span_uuid", "error"), &OpenTelemetry::record_error);
	ClassDB::bind_method(D_METHOD("end_span", "span_uuid"), &OpenTelemetry::end_span);
	ClassDB::bind_method(D_METHOD("set_flush_interval", "interval_ms"), &OpenTelemetry::set_flush_interval);
	ClassDB::bind_method(D_METHOD("set_batch_size", "size"), &OpenTelemetry::set_batch_size);
	ClassDB::bind_method(D_METHOD("record_metric", "name", "value", "unit", "metric_type", "attributes"), &OpenTelemetry::record_metric);
	ClassDB::bind_method(D_METHOD("log_message", "level", "message", "attributes"), &OpenTelemetry::log_message);
	ClassDB::bind_method(D_METHOD("flush_all"), &OpenTelemetry::flush_all);
	ClassDB::bind_method(D_METHOD("shutdown"), &OpenTelemetry::shutdown);

	// Metrics API
	ClassDB::bind_method(D_METHOD("create_counter", "name", "unit", "description"), &OpenTelemetry::create_counter, DEFVAL(""), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("create_gauge", "name", "unit", "description"), &OpenTelemetry::create_gauge, DEFVAL(""), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("create_histogram", "name", "unit", "description"), &OpenTelemetry::create_histogram, DEFVAL(""), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("increment_counter", "counter_id", "value", "attributes"), &OpenTelemetry::increment_counter, DEFVAL(1.0), DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("set_gauge", "gauge_id", "value", "attributes"), &OpenTelemetry::set_gauge, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("record_histogram", "histogram_id", "value", "attributes"), &OpenTelemetry::record_histogram, DEFVAL(Dictionary()));

	// Multi-sink API
	ClassDB::bind_method(D_METHOD("add_sink", "sink_name", "hostname", "headers"), &OpenTelemetry::add_sink, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("remove_sink", "sink_name"), &OpenTelemetry::remove_sink);
	ClassDB::bind_method(D_METHOD("set_sink_enabled", "sink_name", "enabled"), &OpenTelemetry::set_sink_enabled);
	ClassDB::bind_method(D_METHOD("get_sink", "sink_name"), &OpenTelemetry::get_sink);
	ClassDB::bind_method(D_METHOD("list_sinks"), &OpenTelemetry::list_sinks);

	// Direct access to new classes
	ClassDB::bind_method(D_METHOD("get_state"), &OpenTelemetry::get_state);
	ClassDB::bind_method(D_METHOD("get_document"), &OpenTelemetry::get_document);

	BIND_ENUM_CONSTANT(SPAN_KIND_INTERNAL);
	BIND_ENUM_CONSTANT(SPAN_KIND_SERVER);
	BIND_ENUM_CONSTANT(SPAN_KIND_CLIENT);
	BIND_ENUM_CONSTANT(SPAN_KIND_PRODUCER);
	BIND_ENUM_CONSTANT(SPAN_KIND_CONSUMER);

	BIND_ENUM_CONSTANT(STATUS_UNSET);
	BIND_ENUM_CONSTANT(STATUS_OK);
	BIND_ENUM_CONSTANT(STATUS_ERROR);
}

String OpenTelemetry::init_tracer_provider(String p_name, String p_host, Dictionary p_attributes, String p_version, String p_schema_url) {
	tracer_name = p_name;
	hostname = p_host;
	resource_attributes = p_attributes;

	// Update state resource
	if (!p_attributes.is_empty()) {
		Array keys = p_attributes.keys();
		for (int i = 0; i < keys.size(); i++) {
			String key = keys[i];
			state->get_resource()->add_attribute(key, p_attributes[key]);
		}
	}

	// Update scope
	state->get_scope()->set_name(p_name);
	if (!p_version.is_empty()) {
		state->get_scope()->set_version(p_version);
	}

	return "OK";
}

String OpenTelemetry::set_headers(Dictionary p_headers) {
	headers = p_headers;
	return "OK";
}

String OpenTelemetry::generate_trace_id() {
	Ref<Crypto> crypto = Crypto::create();
	PackedByteArray random_bytes = crypto->generate_random_bytes(16);
	StringBuilder hex;
	for (int i = 0; i < random_bytes.size(); i++) {
		hex.append(String::num_int64(random_bytes[i], 16, false).pad_zeros(2));
	}
	return hex.as_string();
}

String OpenTelemetry::generate_span_id() {
	Ref<Crypto> crypto = Crypto::create();
	PackedByteArray random_bytes = crypto->generate_random_bytes(8);
	StringBuilder hex;
	for (int i = 0; i < random_bytes.size(); i++) {
		hex.append(String::num_int64(random_bytes[i], 16, false).pad_zeros(2));
	}
	return hex.as_string();
}

String OpenTelemetry::generate_uuid_v7() {
	uint64_t timestamp_ms = Time::get_singleton()->get_unix_time_from_system() * 1000;
	Ref<Crypto> crypto = Crypto::create();
	PackedByteArray random_bytes = crypto->generate_random_bytes(10);

	PackedByteArray uuid_bytes;
	uuid_bytes.resize(16);
	uint8_t *uuid_ptr = uuid_bytes.ptrw();

	// Initialize all bytes to 0 first
	for (int i = 0; i < 16; i++) {
		uuid_ptr[i] = 0;
	}

	// Timestamp (48 bits) - bytes 0-5
	for (int i = 0; i < 6; i++) {
		uuid_ptr[i] = (timestamp_ms >> (40 - i * 8)) & 0xFF;
	}

	// Version and random - byte 6
	// UUID v7: version 7 (0111) in upper 4 bits of byte 6, lower 4 bits are random
	// Set upper 4 bits to 0111 (7), lower 4 bits to random
	uuid_ptr[6] = (uint8_t)(0x70 | (random_bytes[0] & 0x0F));
	uuid_ptr[7] = random_bytes[1];

	// Variant: 10 in upper 2 bits of byte 8, lower 6 bits are random
	// Set upper 2 bits to 10 (binary), lower 6 bits to random
	uuid_ptr[8] = (uint8_t)(0x80 | (random_bytes[2] & 0x3F));

	for (int i = 9; i < 16; i++) {
		uuid_ptr[i] = random_bytes[i - 6];
	}

	StringBuilder hex;
	for (int i = 0; i < uuid_bytes.size(); i++) {
		// Use num_uint64 to ensure proper unsigned conversion, then pad to 2 hex digits
		String hex_byte = String::num_uint64((uint64_t)uuid_bytes[i], 16, false);
		// Ensure exactly 2 hex characters (pad with leading zero if needed)
		if (hex_byte.length() < 2) {
			hex.append(String("0"));
		}
		hex.append(hex_byte);
	}
	String hex_str = hex.as_string();
	StringBuilder uuid;
	uuid.append(hex_str.substr(0, 8));
	uuid.append(String("-"));
	uuid.append(hex_str.substr(8, 4));
	uuid.append(String("-"));
	uuid.append(hex_str.substr(12, 4));
	uuid.append(String("-"));
	uuid.append(hex_str.substr(16, 4));
	uuid.append(String("-"));
	uuid.append(hex_str.substr(20, 12));
	return uuid.as_string();
}

String OpenTelemetry::start_span(String p_name, SpanKind p_kind, Array p_links, Dictionary p_attributes) {
	if (trace_id.is_empty()) {
		trace_id = generate_trace_id();
	}

	return start_span_with_id(p_name, generate_span_id());
}

String OpenTelemetry::start_span_with_parent(String p_name, String p_parent_span_uuid, SpanKind p_kind, Array p_links, Dictionary p_attributes) {
	return start_span_with_parent_id(p_name, p_parent_span_uuid, generate_span_id());
}

String OpenTelemetry::start_span_with_id(String p_name, String p_span_id) {
	Ref<OpenTelemetrySpan> span;
	span.instantiate();

	span->set_name(p_name);
	span->set_trace_id(trace_id);
	span->set_span_id(p_span_id);
	span->set_kind(OpenTelemetrySpan::SPAN_KIND_INTERNAL);
	span->set_start_time_unix_nano(Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL);

	String uuid = generate_uuid_v7();
	active_spans[uuid] = span;

	return uuid;
}

String OpenTelemetry::start_span_with_parent_id(String p_name, String p_parent_span_uuid, String p_span_id) {
	if (!active_spans.has(p_parent_span_uuid)) {
		return "";
	}

	Ref<OpenTelemetrySpan> parent_span = active_spans[p_parent_span_uuid];

	Ref<OpenTelemetrySpan> span;
	span.instantiate();

	span->set_name(p_name);
	span->set_trace_id(parent_span->get_trace_id());
	span->set_span_id(p_span_id);
	span->set_parent_span_id(parent_span->get_span_id());
	span->set_kind(OpenTelemetrySpan::SPAN_KIND_INTERNAL);
	span->set_start_time_unix_nano(Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL);

	String uuid = generate_uuid_v7();
	active_spans[uuid] = span;

	return uuid;
}

void OpenTelemetry::update_name(String p_span_uuid, String p_name) {
	if (!active_spans.has(p_span_uuid)) {
		return;
	}

	Ref<OpenTelemetrySpan> span = active_spans[p_span_uuid];
	span->set_name(p_name);
}

void OpenTelemetry::add_event(String p_span_uuid, String p_event_name, Dictionary p_attributes, uint64_t p_timestamp) {
	if (!active_spans.has(p_span_uuid)) {
		return;
	}

	Ref<OpenTelemetrySpan> span = active_spans[p_span_uuid];

	uint64_t timestamp = p_timestamp;
	if (timestamp == 0) {
		timestamp = Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL;
	}

	span->add_event(p_event_name, p_attributes, timestamp);
}

void OpenTelemetry::record_exception(String p_span_uuid, String p_message, Dictionary p_attributes) {
	if (!active_spans.has(p_span_uuid)) {
		return;
	}

	Dictionary exception_attrs = p_attributes.duplicate();
	exception_attrs["exception.message"] = p_message;
	exception_attrs["exception.type"] = "Exception";

	add_event(p_span_uuid, "exception", exception_attrs);
}

void OpenTelemetry::set_attributes(String p_span_uuid, Dictionary p_attributes) {
	if (!active_spans.has(p_span_uuid)) {
		return;
	}

	Ref<OpenTelemetrySpan> span = active_spans[p_span_uuid];
	span->set_attributes(p_attributes);
}

void OpenTelemetry::set_status(String p_span_uuid, int p_status_code, String p_description) {
	if (!active_spans.has(p_span_uuid)) {
		return;
	}

	Ref<OpenTelemetrySpan> span = active_spans[p_span_uuid];
	span->set_status_code((OpenTelemetrySpan::StatusCode)p_status_code);

	if (!p_description.is_empty()) {
		span->set_status_message(p_description);
	}
}

void OpenTelemetry::record_error(String p_span_uuid, String p_error) {
	set_status(p_span_uuid, STATUS_ERROR, p_error);
	record_exception(p_span_uuid, p_error);
}

void OpenTelemetry::end_span(String p_span_uuid) {
	if (!active_spans.has(p_span_uuid)) {
		return;
	}

	Ref<OpenTelemetrySpan> span = active_spans[p_span_uuid];
	span->mark_ended();

	// Add to state for export
	state->add_span(span);
	active_spans.erase(p_span_uuid);

	CheckAndFlush();
}

void OpenTelemetry::set_flush_interval(int p_interval_ms) {
	flush_interval_ms = p_interval_ms;
}

void OpenTelemetry::set_batch_size(int p_size) {
	batch_size = p_size;
}

void OpenTelemetry::record_metric(String p_name, float p_value, String p_unit, int p_metric_type, Dictionary p_attributes) {
	Ref<OpenTelemetryMetric> metric;
	metric.instantiate();

	metric->set_name(p_name);
	if (!p_unit.is_empty()) {
		metric->set_unit(p_unit);
	}
	metric->set_metric_type((OpenTelemetryMetric::MetricType)p_metric_type);

	// Create data point
	Dictionary data_point;
	data_point["time_unix_nano"] = Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL;
	data_point["value"] = p_value;
	if (!p_attributes.is_empty()) {
		data_point["attributes"] = p_attributes;
	}

	metric->add_data_point(data_point);

	state->add_metric(metric);
	CheckAndFlush();
}

void OpenTelemetry::log_message(String p_level, String p_message, Dictionary p_attributes) {
	Ref<OpenTelemetryLog> log;
	log.instantiate();

	log->set_body(p_message);
	if (!p_attributes.is_empty()) {
		log->set_attributes(p_attributes);
	}
	log->set_time_unix_nano(Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL);

	// Map log level string to severity
	if (p_level == "TRACE") {
		log->set_severity_number(OpenTelemetryLog::SEVERITY_NUMBER_TRACE);
	} else if (p_level == "DEBUG") {
		log->set_severity_number(OpenTelemetryLog::SEVERITY_NUMBER_DEBUG);
	} else if (p_level == "INFO") {
		log->set_severity_number(OpenTelemetryLog::SEVERITY_NUMBER_INFO);
	} else if (p_level == "WARN") {
		log->set_severity_number(OpenTelemetryLog::SEVERITY_NUMBER_WARN);
	} else if (p_level == "ERROR") {
		log->set_severity_number(OpenTelemetryLog::SEVERITY_NUMBER_ERROR);
	} else if (p_level == "FATAL") {
		log->set_severity_number(OpenTelemetryLog::SEVERITY_NUMBER_FATAL);
	}
	log->set_severity_text(p_level);

	state->add_log(log);
	CheckAndFlush();
}

void OpenTelemetry::CheckAndFlush() {
	uint64_t current_time = Time::get_singleton()->get_unix_time_from_system() * 1000;

	bool should_flush = (state->get_spans().size() >= batch_size) ||
			(current_time - last_flush_time >= (uint64_t)flush_interval_ms);

	if (!should_flush) {
		return;
	}

	FlushAllBufferedData();
	last_flush_time = current_time;
}

Error OpenTelemetry::_send_otlp_request(const String &p_endpoint, const String &p_json_body) {
	if (hostname.is_empty()) {
		return ERR_UNCONFIGURED;
	}

	// Send to default sink
	Error err = _send_to_sink(hostname, headers, p_endpoint, p_json_body);
	if (err != OK) {
		ERR_PRINT("Failed to send to default sink: " + itos(err));
	}

	// Send to additional sinks
	Array sink_names = sinks.keys();
	for (int i = 0; i < sink_names.size(); i++) {
		String sink_name = sink_names[i];
		Dictionary sink = sinks[sink_name];

		if (!sink.has("enabled") || !sink["enabled"]) {
			continue;
		}

		String sink_hostname = sink.get("hostname", "");
		Dictionary sink_headers = sink.get("headers", Dictionary());

		Error sink_err = _send_to_sink(sink_hostname, sink_headers, p_endpoint, p_json_body);
		if (sink_err != OK) {
			ERR_PRINT("Failed to send to sink '" + sink_name + "': " + itos(sink_err));
		}
	}

	return OK;
}

Error OpenTelemetry::_send_to_sink(const String &p_sink_hostname, const Dictionary &p_sink_headers, const String &p_endpoint, const String &p_json_body) {
	if (p_sink_hostname.is_empty()) {
		return ERR_UNCONFIGURED;
	}

	// Parse hostname
	String host = p_sink_hostname;
	int port = 4318;
	bool use_ssl = false;

	if (host.begins_with("https://")) {
		use_ssl = true;
		host = host.substr(8);
	} else if (host.begins_with("http://")) {
		host = host.substr(7);
	}

	int colon_pos = host.find(":");
	if (colon_pos != -1) {
		port = host.substr(colon_pos + 1).to_int();
		host = host.substr(0, colon_pos);
	}

	// Create HTTP client
	Ref<HTTPClient> client = HTTPClient::create();
	Ref<TLSOptions> tls_options;
	if (use_ssl) {
		tls_options = TLSOptions::client();
	}

	Error err = client->connect_to_host(host, port, tls_options);
	if (err != OK) {
		return err;
	}

	// Wait for connection
	while (client->get_status() == HTTPClient::STATUS_CONNECTING || client->get_status() == HTTPClient::STATUS_RESOLVING) {
		client->poll();
		OS::get_singleton()->delay_usec(1000);
	}

	if (client->get_status() != HTTPClient::STATUS_CONNECTED) {
		return FAILED;
	}

	// Prepare headers
	Vector<String> request_headers;
	request_headers.push_back("Content-Type: application/json");

	Array header_keys = p_sink_headers.keys();
	for (int i = 0; i < header_keys.size(); i++) {
		String key = header_keys[i];
		String value = p_sink_headers[key];
		request_headers.push_back(key + ": " + value);
	}

	// Send request
	CharString body_utf8 = p_json_body.utf8();
	err = client->request(HTTPClient::METHOD_POST, p_endpoint, request_headers, (const uint8_t *)body_utf8.get_data(), body_utf8.length());
	if (err != OK) {
		return err;
	}

	// Wait for response
	while (client->get_status() == HTTPClient::STATUS_REQUESTING) {
		client->poll();
		OS::get_singleton()->delay_usec(1000);
	}

	if (client->get_status() != HTTPClient::STATUS_BODY && client->get_status() != HTTPClient::STATUS_CONNECTED) {
		return FAILED;
	}

	return OK;
}

void OpenTelemetry::FlushAllBufferedData() {
	// Move any ended active spans to state
	Array keys = active_spans.keys();
	for (int i = 0; i < keys.size(); i++) {
		Ref<OpenTelemetrySpan> span = active_spans[keys[i]];
		if (!span.is_valid() || !span->is_ended()) {
			continue;
		}

		state->add_span(span);
		active_spans.erase(keys[i]);
	}

	// Use OpenTelemetryDocument for serialization
	if (state->get_spans().size() > 0) {
		String json = document->serialize_traces(state);
		_send_otlp_request("/v1/traces", json);
		state->clear_spans();
	}

	if (state->get_metrics().size() > 0) {
		String json = document->serialize_metrics(state);
		_send_otlp_request("/v1/metrics", json);
		state->clear_metrics();
	}

	if (state->get_logs().size() > 0) {
		String json = document->serialize_logs(state);
		_send_otlp_request("/v1/logs", json);
		state->clear_logs();
	}
}

String OpenTelemetry::add_sink(String p_sink_name, String p_hostname, Dictionary p_headers) {
	if (p_sink_name.is_empty() || p_hostname.is_empty()) {
		return "ERROR: Sink name and hostname are required";
	}

	Dictionary sink;
	sink["hostname"] = p_hostname;
	sink["headers"] = p_headers;
	sink["enabled"] = true;

	sinks[p_sink_name] = sink;
	return "OK";
}

String OpenTelemetry::remove_sink(String p_sink_name) {
	if (!sinks.has(p_sink_name)) {
		return "ERROR: Sink not found";
	}

	sinks.erase(p_sink_name);
	return "OK";
}

String OpenTelemetry::set_sink_enabled(String p_sink_name, bool p_enabled) {
	if (!sinks.has(p_sink_name)) {
		return "ERROR: Sink not found";
	}

	Dictionary sink = sinks[p_sink_name];
	sink["enabled"] = p_enabled;
	sinks[p_sink_name] = sink;
	return "OK";
}

Dictionary OpenTelemetry::get_sink(String p_sink_name) {
	if (!sinks.has(p_sink_name)) {
		return Dictionary();
	}

	return sinks[p_sink_name];
}

Array OpenTelemetry::list_sinks() {
	return sinks.keys();
}

void OpenTelemetry::flush_all() {
	FlushAllBufferedData();
}

String OpenTelemetry::shutdown() {
	FlushAllBufferedData();
	active_spans.clear();
	state->clear_all();
	return "OK";
}

// Metrics API

String OpenTelemetry::create_counter(String p_name, String p_unit, String p_description) {
	return generate_uuid_v7();
}

String OpenTelemetry::create_gauge(String p_name, String p_unit, String p_description) {
	return generate_uuid_v7();
}

String OpenTelemetry::create_histogram(String p_name, String p_unit, String p_description) {
	return generate_uuid_v7();
}

void OpenTelemetry::increment_counter(String p_counter_id, float p_value, Dictionary p_attributes) {
	record_metric("counter_" + p_counter_id, p_value, "", 0, p_attributes);
}

void OpenTelemetry::set_gauge(String p_gauge_id, float p_value, Dictionary p_attributes) {
	record_metric("gauge_" + p_gauge_id, p_value, "", 1, p_attributes);
}

void OpenTelemetry::record_histogram(String p_histogram_id, float p_value, Dictionary p_attributes) {
	record_metric("histogram_" + p_histogram_id, p_value, "", 2, p_attributes);
}
