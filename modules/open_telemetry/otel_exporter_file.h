/**************************************************************************/
/*  otel_exporter_file.h                                                  */
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
#include "otel_document.h"
#include "otel_state.h"

// File-based OTLP exporter - writes telemetry data to JSON files
class OpenTelemetryExporterFile : public Resource {
	GDCLASS(OpenTelemetryExporterFile, Resource);

public:
	enum ExportMode {
		EXPORT_MODE_APPEND = 0, // Append to existing file
		EXPORT_MODE_OVERWRITE = 1, // Overwrite file each time
		EXPORT_MODE_TIMESTAMPED = 2 // Create timestamped files
	};

private:
	String output_directory;
	String filename_prefix;
	ExportMode mode = EXPORT_MODE_TIMESTAMPED;
	Ref<OpenTelemetryDocument> document;
	bool pretty_print = true;

protected:
	static void _bind_methods();

public:
	OpenTelemetryExporterFile();

	// Configuration
	void set_output_directory(const String &p_directory);
	String get_output_directory() const;

	void set_filename_prefix(const String &p_prefix);
	String get_filename_prefix() const;

	void set_mode(ExportMode p_mode);
	ExportMode get_mode() const;

	void set_pretty_print(bool p_enabled);
	bool get_pretty_print() const;

	// Export operations
	Error export_traces(Ref<OpenTelemetryState> p_state);
	Error export_metrics(Ref<OpenTelemetryState> p_state);
	Error export_logs(Ref<OpenTelemetryState> p_state);

	// Helper methods
	String generate_filename(const String &p_type) const;
	Error ensure_directory_exists();
};

VARIANT_ENUM_CAST(OpenTelemetryExporterFile::ExportMode);
