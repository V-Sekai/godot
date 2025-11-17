/**************************************************************************/
/*  otel_reflector.cpp                                                    */
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

#include "otel_reflector.h"

#include "otel_document.h"
#include "otel_state.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"

void OpenTelemetryReflector::_bind_methods() {
	// Load from file
	ClassDB::bind_method(D_METHOD("load_traces_from_file", "path"), &OpenTelemetryReflector::load_traces_from_file);
	ClassDB::bind_method(D_METHOD("load_metrics_from_file", "path"), &OpenTelemetryReflector::load_metrics_from_file);
	ClassDB::bind_method(D_METHOD("load_logs_from_file", "path"), &OpenTelemetryReflector::load_logs_from_file);

	// Load from JSON string
	ClassDB::bind_method(D_METHOD("load_traces_from_json", "json"), &OpenTelemetryReflector::load_traces_from_json);
	ClassDB::bind_method(D_METHOD("load_metrics_from_json", "json"), &OpenTelemetryReflector::load_metrics_from_json);
	ClassDB::bind_method(D_METHOD("load_logs_from_json", "json"), &OpenTelemetryReflector::load_logs_from_json);

	// Load and merge
	ClassDB::bind_method(D_METHOD("load_and_merge_traces", "paths"), &OpenTelemetryReflector::load_and_merge_traces);
	ClassDB::bind_method(D_METHOD("load_and_merge_metrics", "paths"), &OpenTelemetryReflector::load_and_merge_metrics);
	ClassDB::bind_method(D_METHOD("load_and_merge_logs", "paths"), &OpenTelemetryReflector::load_and_merge_logs);

	// Utility
	ClassDB::bind_method(D_METHOD("find_otlp_files", "directory", "pattern"), &OpenTelemetryReflector::find_otlp_files, DEFVAL("*.json"));
	ClassDB::bind_method(D_METHOD("get_document"), &OpenTelemetryReflector::get_document);
}

OpenTelemetryReflector::OpenTelemetryReflector() {
	document.instantiate();
}

// Load OTLP JSON from file
Ref<OpenTelemetryState> OpenTelemetryReflector::load_traces_from_file(const String &p_path) {
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	if (file.is_null()) {
		ERR_PRINT("OpenTelemetryReflector: Failed to open file: " + p_path);
		return Ref<OpenTelemetryState>();
	}

	String json = file->get_as_text();
	file.unref();

	return load_traces_from_json(json);
}

Ref<OpenTelemetryState> OpenTelemetryReflector::load_metrics_from_file(const String &p_path) {
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	if (file.is_null()) {
		ERR_PRINT("OpenTelemetryReflector: Failed to open file: " + p_path);
		return Ref<OpenTelemetryState>();
	}

	String json = file->get_as_text();
	file.unref();

	return load_metrics_from_json(json);
}

Ref<OpenTelemetryState> OpenTelemetryReflector::load_logs_from_file(const String &p_path) {
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	if (file.is_null()) {
		ERR_PRINT("OpenTelemetryReflector: Failed to open file: " + p_path);
		return Ref<OpenTelemetryState>();
	}

	String json = file->get_as_text();
	file.unref();

	return load_logs_from_json(json);
}

// Load OTLP JSON from string (delegates to OpenTelemetryDocument)
Ref<OpenTelemetryState> OpenTelemetryReflector::load_traces_from_json(const String &p_json) {
	return document->deserialize_traces(p_json);
}

Ref<OpenTelemetryState> OpenTelemetryReflector::load_metrics_from_json(const String &p_json) {
	return document->deserialize_metrics(p_json);
}

Ref<OpenTelemetryState> OpenTelemetryReflector::load_logs_from_json(const String &p_json) {
	return document->deserialize_logs(p_json);
}

// Load multiple files and merge
Ref<OpenTelemetryState> OpenTelemetryReflector::load_and_merge_traces(const PackedStringArray &p_paths) {
	Ref<OpenTelemetryState> merged_state;
	merged_state.instantiate();

	for (int i = 0; i < p_paths.size(); i++) {
		Ref<OpenTelemetryState> state = load_traces_from_file(p_paths[i]);
		if (state.is_valid()) {
			// Merge spans
			TypedArray<OpenTelemetrySpan> spans = state->get_spans();
			for (int j = 0; j < spans.size(); j++) {
				merged_state->add_span(spans[j]);
			}

			// Copy resource and scope from first file
			if (i == 0) {
				merged_state->set_resource(state->get_resource());
				merged_state->set_scope(state->get_scope());
			}
		}
	}

	return merged_state;
}

Ref<OpenTelemetryState> OpenTelemetryReflector::load_and_merge_metrics(const PackedStringArray &p_paths) {
	Ref<OpenTelemetryState> merged_state;
	merged_state.instantiate();

	for (int i = 0; i < p_paths.size(); i++) {
		Ref<OpenTelemetryState> state = load_metrics_from_file(p_paths[i]);
		if (state.is_valid()) {
			// Merge metrics
			TypedArray<OpenTelemetryMetric> metrics = state->get_metrics();
			for (int j = 0; j < metrics.size(); j++) {
				merged_state->add_metric(metrics[j]);
			}

			// Copy resource and scope from first file
			if (i == 0) {
				merged_state->set_resource(state->get_resource());
				merged_state->set_scope(state->get_scope());
			}
		}
	}

	return merged_state;
}

Ref<OpenTelemetryState> OpenTelemetryReflector::load_and_merge_logs(const PackedStringArray &p_paths) {
	Ref<OpenTelemetryState> merged_state;
	merged_state.instantiate();

	for (int i = 0; i < p_paths.size(); i++) {
		Ref<OpenTelemetryState> state = load_logs_from_file(p_paths[i]);
		if (state.is_valid()) {
			// Merge logs
			TypedArray<OpenTelemetryLog> logs = state->get_logs();
			for (int j = 0; j < logs.size(); j++) {
				merged_state->add_log(logs[j]);
			}

			// Copy resource and scope from first file
			if (i == 0) {
				merged_state->set_resource(state->get_resource());
				merged_state->set_scope(state->get_scope());
			}
		}
	}

	return merged_state;
}

// Utility: Find all OTLP JSON files in directory
PackedStringArray OpenTelemetryReflector::find_otlp_files(const String &p_directory, const String &p_pattern) {
	PackedStringArray result;

	Ref<DirAccess> dir = DirAccess::open(p_directory);
	if (dir.is_null()) {
		ERR_PRINT("OpenTelemetryReflector: Failed to open directory: " + p_directory);
		return result;
	}

	dir->list_dir_begin();
	String file_name = dir->get_next();

	while (!file_name.is_empty()) {
		if (!dir->current_is_dir()) {
			// Check if file matches pattern
			if (p_pattern == "*.json" && file_name.ends_with(".json")) {
				result.push_back(p_directory.path_join(file_name));
			} else if (file_name.match(p_pattern)) {
				result.push_back(p_directory.path_join(file_name));
			}
		}
		file_name = dir->get_next();
	}

	dir->list_dir_end();
	return result;
}

Ref<OpenTelemetryDocument> OpenTelemetryReflector::get_document() const {
	return document;
}
