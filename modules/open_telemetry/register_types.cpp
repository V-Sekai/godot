/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "open_telemetry.h"
#include "open_telemetry_logger.h"

// New structure classes
#include "otel_document.h"
#include "otel_exporter_file.h"
#include "otel_reflector.h"
#include "otel_state.h"
#include "structures/otel_log.h"
#include "structures/otel_metric.h"
#include "structures/otel_resource.h"
#include "structures/otel_scope.h"
#include "structures/otel_span.h"

static OpenTelemetry *global_otel_instance = nullptr;
static OpenTelemetryLogger *global_otel_logger = nullptr;

void initialize_open_telemetry_module(ModuleInitializationLevel p_level) {
	// Handle server initialization
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		// Register project settings for exported games
		GLOBAL_DEF("opentelemetry/enabled", false);
		GLOBAL_DEF("opentelemetry/endpoint", "");
		GLOBAL_DEF("opentelemetry/service_name", "");

		ProjectSettings::get_singleton()->set_custom_property_info(
				PropertyInfo(Variant::BOOL, "opentelemetry/enabled"));
		ProjectSettings::get_singleton()->set_custom_property_info(
				PropertyInfo(Variant::STRING, "opentelemetry/endpoint"));
		ProjectSettings::get_singleton()->set_custom_property_info(
				PropertyInfo(Variant::STRING, "opentelemetry/service_name"));

		bool should_initialize = false;
		String endpoint = "";
		String service_name = "";

#ifdef TOOLS_ENABLED
		// Editor: Check if editor mode and telemetry enabled
		if (!Engine::get_singleton()->is_editor_hint()) {
			return; // Not in editor, skip initialization
		}

		bool editor_telemetry_enabled = GLOBAL_DEF("opentelemetry/editor_enabled", true);
		ProjectSettings::get_singleton()->set_custom_property_info(
				PropertyInfo(Variant::BOOL, "opentelemetry/editor_enabled"));

		if (!editor_telemetry_enabled) {
			return; // Telemetry disabled in editor
		}

		should_initialize = true;
		endpoint = "http://localhost:4318";
		service_name = "godot_editor";
#else
		// Exported game: Check project settings (opt-in)
		if (!GLOBAL_GET("opentelemetry/enabled")) {
			return; // Telemetry not enabled in exported game
		}

		endpoint = GLOBAL_GET("opentelemetry/endpoint");
		service_name = GLOBAL_GET("opentelemetry/service_name");

		if (endpoint.is_empty()) {
			return; // No endpoint configured
		}

		should_initialize = true;

		// Use default service name if not provided
		if (service_name.is_empty()) {
			service_name = "godot_game";
		}
#endif

		if (!should_initialize) {
			return; // Should not happen, but safety check
		}

		// Initialize OpenTelemetry instance
		global_otel_instance = memnew(OpenTelemetry);

		// Initialize with configuration
		Dictionary resource_attrs;
		resource_attrs["service.name"] = service_name;
		String init_result = global_otel_instance->init_tracer_provider("godot_tracer", endpoint, resource_attrs);
		if (init_result != "OK") {
			ERR_PRINT("OpenTelemetry: Failed to initialize tracer provider: " + init_result);
		}

#ifdef TOOLS_ENABLED
		// Create and register custom logger in editor
		global_otel_logger = memnew(OpenTelemetryLogger(global_otel_instance));
		OS::get_singleton()->add_logger(global_otel_logger);
#endif
		return;
	}

	// Handle scene initialization
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	// Register main API classes
	GDREGISTER_CLASS(OpenTelemetryTracerProvider);
	GDREGISTER_CLASS(OpenTelemetry);

	// Register new structure classes (GLTFDocument pattern)
	GDREGISTER_CLASS(OTelSpan);
	GDREGISTER_CLASS(OTelResource);
	GDREGISTER_CLASS(OTelScope);
	GDREGISTER_CLASS(OTelMetric);
	GDREGISTER_CLASS(OTelLog);
	GDREGISTER_CLASS(OTelDocument);
	GDREGISTER_CLASS(OTelState);
	GDREGISTER_CLASS(OTelExporterFile);
	GDREGISTER_CLASS(OTelReflector);
}

void uninitialize_open_telemetry_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		// Clean up logger and OpenTelemetry instance
		if (global_otel_logger) {
			memdelete(global_otel_logger);
			global_otel_logger = nullptr;
		}

		if (global_otel_instance) {
			String shutdown_result = global_otel_instance->shutdown();
			if (shutdown_result != "OK") {
				ERR_PRINT("OpenTelemetry: Failed to shutdown cleanly: " + shutdown_result);
			}
			memdelete(global_otel_instance);
			global_otel_instance = nullptr;
		}
	}

	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	// Nothing to do here for now
}
