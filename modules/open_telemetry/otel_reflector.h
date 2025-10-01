/**************************************************************************/
/*  otel_reflector.h                                                      */
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

class OTelDocument;
class OTelState;

// OTelReflector loads OTLP JSON data (from Python/external sources) into Godot
// Pairs with OTelDocument serialization for bidirectional data flow
class OTelReflector : public Resource {
	GDCLASS(OTelReflector, Resource);

private:
	Ref<OTelDocument> document;

protected:
	static void _bind_methods();

public:
	OTelReflector();

	// Load OTLP JSON from file
	Ref<OTelState> load_traces_from_file(const String &p_path);
	Ref<OTelState> load_metrics_from_file(const String &p_path);
	Ref<OTelState> load_logs_from_file(const String &p_path);

	// Load OTLP JSON from string
	Ref<OTelState> load_traces_from_json(const String &p_json);
	Ref<OTelState> load_metrics_from_json(const String &p_json);
	Ref<OTelState> load_logs_from_json(const String &p_json);

	// Load multiple files and merge
	Ref<OTelState> load_and_merge_traces(const PackedStringArray &p_paths);
	Ref<OTelState> load_and_merge_metrics(const PackedStringArray &p_paths);
	Ref<OTelState> load_and_merge_logs(const PackedStringArray &p_paths);

	// Utility: List all files in directory
	PackedStringArray find_otlp_files(const String &p_directory, const String &p_pattern = "*.json");

	// Get document instance
	Ref<OTelDocument> get_document() const;
};
