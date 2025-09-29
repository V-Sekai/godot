/**************************************************************************/
/*  test_open_telemetry.h                                                 */
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

#include "tests/test_macros.h"

#ifdef TOOLS_ENABLED

#include "core/crypto/crypto.h"
#include "core/io/json.h"
#include "core/os/time.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"

#include "modules/open_telemetry/open_telemetry.h"

namespace TestOpenTelemetry {

TEST_CASE("[OpenTelemetry] Status creation and functionality") {
	// Test Status creation with default values
	Ref<godot::Status> status = memnew(godot::Status);

	CHECK(status->get_status_code() == godot::STATUS_UNSET);
	CHECK(status->get_description() == "");

	// Test Status creation with parameters
	Ref<godot::Status> error_status = memnew(godot::Status(godot::STATUS_ERROR, "Test error message"));

	CHECK(error_status->get_status_code() == godot::STATUS_ERROR);
	CHECK(error_status->get_description() == "Test error message");

	// Test setters
	status->set_status_code(godot::STATUS_OK);
	status->set_description("All good");

	CHECK(status->get_status_code() == godot::STATUS_OK);
	CHECK(status->get_description() == "All good");
}

TEST_CASE("[OpenTelemetry] SpanContext creation and validation") {
	// Test empty SpanContext
	Ref<godot::SpanContext> context = memnew(godot::SpanContext);

	CHECK_FALSE(context->is_valid());
	CHECK_FALSE(context->is_remote());

	// Test SpanContext with valid IDs
	String trace_id = "0123456789abcdef0123456789abcdef";
	String span_id = "0123456789abcdef";

	Ref<godot::SpanContext> valid_context = memnew(godot::SpanContext(trace_id, span_id, false, Dictionary()));

	CHECK(valid_context->is_valid());
	CHECK(valid_context->get_trace_id() == trace_id);
	CHECK(valid_context->get_span_id() == span_id);
	CHECK(valid_context->get_trace_id_hex() == trace_id);
	CHECK(valid_context->get_span_id_hex() == span_id);
	CHECK_FALSE(valid_context->is_remote());

	// Test remote flag
	Ref<godot::SpanContext> remote_context = memnew(godot::SpanContext(trace_id, span_id, true, Dictionary()));
	CHECK(remote_context->is_remote());
}

TEST_CASE("[OpenTelemetry] SpanContext binary conversion") {
	String trace_id = "0123456789abcdef0123456789abcdef";
	String span_id = "0123456789abcdef";

	Ref<godot::SpanContext> context = memnew(godot::SpanContext(trace_id, span_id, false, Dictionary()));

	// Test trace ID binary conversion
	PackedByteArray trace_binary = context->get_trace_id_binary();
	CHECK(trace_binary.size() == 16);
	CHECK(trace_binary[0] == 0x01);
	CHECK(trace_binary[1] == 0x23);
	CHECK(trace_binary[2] == 0x45);
	CHECK(trace_binary[3] == 0x67);

	// Test span ID binary conversion
	PackedByteArray span_binary = context->get_span_id_binary();
	CHECK(span_binary.size() == 8);
	CHECK(span_binary[0] == 0x01);
	CHECK(span_binary[1] == 0x23);
	CHECK(span_binary[2] == 0x45);
	CHECK(span_binary[3] == 0x67);
}

TEST_CASE("[OpenTelemetry] Link creation and functionality") {
	String trace_id = "0123456789abcdef0123456789abcdef";
	String span_id = "0123456789abcdef";

	Ref<godot::SpanContext> context = memnew(godot::SpanContext(trace_id, span_id, false, Dictionary()));

	Dictionary attributes;
	attributes["link.type"] = "test";
	attributes["link.id"] = 123;

	Ref<godot::Link> link = memnew(godot::Link(context, attributes));

	CHECK(link->get_span_context() == context);
	CHECK(link->get_attributes()["link.type"] == "test");
	CHECK(int(link->get_attributes()["link.id"]) == 123);
}

TEST_CASE("[OpenTelemetry] Span creation and basic operations") {
	String trace_id = "0123456789abcdef0123456789abcdef";
	String span_id = "0123456789abcdef";

	Ref<godot::SpanContext> context = memnew(godot::SpanContext(trace_id, span_id, false, Dictionary()));

	Ref<godot::Span> span = memnew(godot::Span("test-span", context, godot::SPAN_KIND_CLIENT));

	CHECK(span->get_name() == "test-span");
	CHECK(span->get_context() == context);
	CHECK(span->get_kind() == godot::SPAN_KIND_CLIENT);
	CHECK(span->is_recording());
	CHECK_FALSE(span->is_ended());
	CHECK(span->get_start_time() > 0);
	CHECK(span->get_end_time() == 0);
}

TEST_CASE("[OpenTelemetry] Span attribute management") {
	String trace_id = "0123456789abcdef0123456789abcdef";
	String span_id = "0123456789abcdef";

	Ref<godot::SpanContext> context = memnew(godot::SpanContext(trace_id, span_id, false, Dictionary()));

	Ref<godot::Span> span = memnew(godot::Span("test-span", context, godot::SPAN_KIND_INTERNAL));

	// Test setting individual attributes
	span->set_attribute("key1", "value1");
	span->set_attribute("key2", 42);
	span->set_attribute("key3", true);

	Dictionary attributes = span->get_attributes();
	CHECK(attributes["key1"] == "value1");
	CHECK(int(attributes["key2"]) == 42);
	CHECK(bool(attributes["key3"]) == true);

	// Test setting multiple attributes
	Dictionary bulk_attrs;
	bulk_attrs["bulk1"] = "bulk_value1";
	bulk_attrs["bulk2"] = 99;

	span->set_attributes(bulk_attrs);

	attributes = span->get_attributes();
	CHECK(attributes["bulk1"] == "bulk_value1");
	CHECK(int(attributes["bulk2"]) == 99);
	// Previous attributes should still exist
	CHECK(attributes["key1"] == "value1");
}

TEST_CASE("[OpenTelemetry] Span lifecycle operations") {
	String trace_id = "0123456789abcdef0123456789abcdef";
	String span_id = "0123456789abcdef";

	Ref<godot::SpanContext> context = memnew(godot::SpanContext(trace_id, span_id, false, Dictionary()));

	Ref<godot::Span> span = memnew(godot::Span("test-span", context, godot::SPAN_KIND_INTERNAL));

	uint64_t start_time = span->get_start_time();
	CHECK(start_time > 0);
	CHECK(span->is_recording());
	CHECK_FALSE(span->is_ended());

	// Update name
	span->update_name("updated-span");
	CHECK(span->get_name() == "updated-span");

	// End span
	uint64_t before_end = Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL;
	span->end();
	uint64_t after_end = Time::get_singleton()->get_unix_time_from_system() * 1000000000ULL;

	CHECK(span->is_ended());
	CHECK_FALSE(span->is_recording());
	CHECK(span->get_end_time() >= before_end);
	CHECK(span->get_end_time() <= after_end);

	// Operations after end should be ignored
	span->set_attribute("ignored", "value");
	CHECK_FALSE(span->get_attributes().has("ignored"));
}

TEST_CASE("[OpenTelemetry] Tracer creation and span generation") {
	Ref<godot::Tracer> tracer = memnew(godot::Tracer("test-tracer", "1.0.0", "http://example.com/schema", Dictionary()));

	CHECK(tracer->get_name() == "test-tracer");
	CHECK(tracer->get_version() == "1.0.0");
	CHECK(tracer->get_schema_url() == "http://example.com/schema");
	CHECK(tracer->enabled());

	// Test span creation
	Dictionary span_attrs;
	span_attrs["test.key"] = "test.value";

	Ref<godot::Span> span = tracer->start_span("test-operation", godot::SPAN_KIND_SERVER, Ref<godot::SpanContext>(), span_attrs, Array());

	CHECK(span->get_name() == "test-operation");
	CHECK(span->get_kind() == godot::SPAN_KIND_SERVER);
	CHECK(span->get_attributes()["test.key"] == "test.value");
	CHECK(span->get_context()->is_valid());
}

TEST_CASE("[OpenTelemetry] TracerProvider functionality") {
	Dictionary resource_attrs;
	resource_attrs["service.name"] = "test-service";
	resource_attrs["service.version"] = "1.0.0";

	Ref<godot::TracerProvider> provider = memnew(godot::TracerProvider(resource_attrs));

	CHECK(provider->get_resource_attributes()["service.name"] == "test-service");

	// Test tracer creation
	Ref<godot::Tracer> tracer1 = provider->get_tracer("tracer1", "1.0", "http://example.com", Dictionary());
	Ref<godot::Tracer> tracer2 = provider->get_tracer("tracer1", "1.0", "http://example.com", Dictionary());

	// Same parameters should return the same tracer instance
	CHECK(tracer1 == tracer2);

	// Different parameters should return different tracer
	Ref<godot::Tracer> tracer3 = provider->get_tracer("tracer2", "1.0", "http://example.com", Dictionary());
	CHECK(tracer1 != tracer3);
}

TEST_CASE("[OpenTelemetry] UUID v7 generation") {
	Ref<godot::OpenTelemetry> otel = memnew(godot::OpenTelemetry);

	String uuid1 = otel->generate_uuid_v7();
	String uuid2 = otel->generate_uuid_v7();

	// UUIDs should be different
	CHECK(uuid1 != uuid2);

	// UUIDs should have correct format (36 characters with hyphens)
	CHECK(uuid1.length() == 36);
	CHECK(uuid2.length() == 36);
	CHECK(uuid1.substr(8, 1) == "-");
	CHECK(uuid1.substr(13, 1) == "-");
	CHECK(uuid1.substr(18, 1) == "-");
	CHECK(uuid1.substr(23, 1) == "-");

	// Version should be 7 (character at position 14)
	CHECK(uuid1.substr(14, 1) == "7");
	CHECK(uuid2.substr(14, 1) == "7");
}

} //namespace TestOpenTelemetry

#endif // TOOLS_ENABLED
