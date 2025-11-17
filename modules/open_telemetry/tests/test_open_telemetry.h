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

// TODO: Rewrite tests to use actual OpenTelemetry API classes:
// - OpenTelemetrySpan (not godot::Span)
// - OpenTelemetryTracer (not godot::Tracer)
// - OpenTelemetryTracerProvider (not godot::TracerProvider)
// The current tests reference classes that don't exist (Status, SpanContext, Link, etc.)

TEST_CASE("[OpenTelemetry] Basic span creation") {
	Ref<OpenTelemetrySpan> span = memnew(OpenTelemetrySpan);

	CHECK(span->get_name() == "");
	CHECK(span->get_trace_id().length() == 32);
	CHECK(span->get_span_id().length() == 16);
	CHECK(span->get_kind() == OpenTelemetrySpan::SPAN_KIND_INTERNAL);
	CHECK(span->get_status_code() == OpenTelemetrySpan::STATUS_CODE_UNSET);
	CHECK_FALSE(span->is_ended());
}

TEST_CASE("[OpenTelemetry] Span attribute management") {
	Ref<OpenTelemetrySpan> span = memnew(OpenTelemetrySpan);

	// Test setting individual attributes
	span->add_attribute("key1", "value1");
	span->add_attribute("key2", 42);
	span->add_attribute("key3", true);

	Dictionary attributes = span->get_attributes();
	CHECK(attributes["key1"] == "value1");
	CHECK(int(attributes["key2"]) == 42);
	CHECK(bool(attributes["key3"]) == true);
}

TEST_CASE("[OpenTelemetry] Tracer creation") {
	Ref<OpenTelemetryTracer> tracer = memnew(OpenTelemetryTracer("test-tracer", "1.0.0", "http://example.com/schema", Dictionary()));

	CHECK(tracer->get_name() == "test-tracer");
	CHECK(tracer->get_version() == "1.0.0");
	CHECK(tracer->get_schema_url() == "http://example.com/schema");
	CHECK(tracer->enabled());
}

TEST_CASE("[OpenTelemetry] TracerProvider functionality") {
	Dictionary resource_attrs;
	resource_attrs["service.name"] = "test-service";
	resource_attrs["service.version"] = "1.0.0";

	Ref<OpenTelemetryTracerProvider> provider = memnew(OpenTelemetryTracerProvider(resource_attrs));

	CHECK(provider->get_resource_attributes()["service.name"] == "test-service");

	// Test tracer creation
	Ref<OpenTelemetryTracer> tracer1 = provider->get_tracer("tracer1", "1.0", "http://example.com", Dictionary());
	Ref<OpenTelemetryTracer> tracer2 = provider->get_tracer("tracer1", "1.0", "http://example.com", Dictionary());

	// Same parameters should return the same tracer instance
	CHECK(tracer1 == tracer2);

	// Different parameters should return different tracer
	Ref<OpenTelemetryTracer> tracer3 = provider->get_tracer("tracer2", "1.0", "http://example.com", Dictionary());
	CHECK(tracer1 != tracer3);
}

TEST_CASE("[OpenTelemetry] UUID v7 generation") {
	Ref<OpenTelemetry> otel = memnew(OpenTelemetry);

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
