/**************************************************************************/
/*  otel_exporter_file.cpp                                                */
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

#include "otel_exporter_file.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/os/time.h"

void OpenTelemetryExporterFile::_bind_methods() {
	// Configuration
	ClassDB::bind_method(D_METHOD("set_output_directory", "directory"), &OpenTelemetryExporterFile::set_output_directory);
	ClassDB::bind_method(D_METHOD("get_output_directory"), &OpenTelemetryExporterFile::get_output_directory);
	ClassDB::bind_method(D_METHOD("set_filename_prefix", "prefix"), &OpenTelemetryExporterFile::set_filename_prefix);
	ClassDB::bind_method(D_METHOD("get_filename_prefix"), &OpenTelemetryExporterFile::get_filename_prefix);
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &OpenTelemetryExporterFile::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &OpenTelemetryExporterFile::get_mode);
	ClassDB::bind_method(D_METHOD("set_pretty_print", "enabled"), &OpenTelemetryExporterFile::set_pretty_print);
	ClassDB::bind_method(D_METHOD("get_pretty_print"), &OpenTelemetryExporterFile::get_pretty_print);

	// Export operations
	ClassDB::bind_method(D_METHOD("export_traces", "state"), &OpenTelemetryExporterFile::export_traces);
	ClassDB::bind_method(D_METHOD("export_metrics", "state"), &OpenTelemetryExporterFile::export_metrics);
	ClassDB::bind_method(D_METHOD("export_logs", "state"), &OpenTelemetryExporterFile::export_logs);

	// Helper methods
	ClassDB::bind_method(D_METHOD("generate_filename", "type"), &OpenTelemetryExporterFile::generate_filename);
	ClassDB::bind_method(D_METHOD("ensure_directory_exists"), &OpenTelemetryExporterFile::ensure_directory_exists);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "output_directory", PROPERTY_HINT_DIR), "set_output_directory", "get_output_directory");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "filename_prefix"), "set_filename_prefix", "get_filename_prefix");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Append,Overwrite,Timestamped"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pretty_print"), "set_pretty_print", "get_pretty_print");

	BIND_ENUM_CONSTANT(EXPORT_MODE_APPEND);
	BIND_ENUM_CONSTANT(EXPORT_MODE_OVERWRITE);
	BIND_ENUM_CONSTANT(EXPORT_MODE_TIMESTAMPED);
}

OpenTelemetryExporterFile::OpenTelemetryExporterFile() {
	document.instantiate();
	output_directory = "user://otel_traces";
	filename_prefix = "otel";
}

// Configuration
void OpenTelemetryExporterFile::set_output_directory(const String &p_directory) {
	output_directory = p_directory;
}

String OpenTelemetryExporterFile::get_output_directory() const {
	return output_directory;
}

void OpenTelemetryExporterFile::set_filename_prefix(const String &p_prefix) {
	filename_prefix = p_prefix;
}

String OpenTelemetryExporterFile::get_filename_prefix() const {
	return filename_prefix;
}

void OpenTelemetryExporterFile::set_mode(ExportMode p_mode) {
	mode = p_mode;
}

OpenTelemetryExporterFile::ExportMode OpenTelemetryExporterFile::get_mode() const {
	return mode;
}

void OpenTelemetryExporterFile::set_pretty_print(bool p_enabled) {
	pretty_print = p_enabled;
}

bool OpenTelemetryExporterFile::get_pretty_print() const {
	return pretty_print;
}

// Helper methods
String OpenTelemetryExporterFile::generate_filename(const String &p_type) const {
	String filename;

	switch (mode) {
		case EXPORT_MODE_APPEND:
		case EXPORT_MODE_OVERWRITE:
			filename = vformat("%s_%s.json", filename_prefix, p_type);
			break;
		case EXPORT_MODE_TIMESTAMPED: {
			Dictionary time_dict = Time::get_singleton()->get_datetime_dict_from_system();
			filename = vformat("%s_%s_%04d%02d%02d_%02d%02d%02d.json",
					filename_prefix,
					p_type,
					(int)time_dict["year"],
					(int)time_dict["month"],
					(int)time_dict["day"],
					(int)time_dict["hour"],
					(int)time_dict["minute"],
					(int)time_dict["second"]);
			break;
		}
	}

	return output_directory.path_join(filename);
}

Error OpenTelemetryExporterFile::ensure_directory_exists() {
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_USERDATA);
	if (dir.is_null()) {
		return ERR_CANT_CREATE;
	}

	if (!dir->dir_exists(output_directory)) {
		Error err = dir->make_dir_recursive(output_directory);
		if (err != OK) {
			ERR_PRINT(vformat("OpenTelemetryExporterFile: Failed to create directory: %s", output_directory));
			return err;
		}
	}

	return OK;
}

// Export operations
Error OpenTelemetryExporterFile::export_traces(Ref<OpenTelemetryState> p_state) {
	if (p_state.is_null()) {
		return ERR_INVALID_PARAMETER;
	}

	Error err = ensure_directory_exists();
	if (err != OK) {
		return err;
	}

	// Serialize using OpenTelemetryDocument
	String json_data = document->serialize_traces(p_state);

	// Pretty print if enabled
	if (pretty_print) {
		JSON json_parser;
		Error parse_err = json_parser.parse(json_data);
		if (parse_err == OK) {
			json_data = JSON::stringify(json_parser.get_data(), "\t", false);
		}
	}

	// Generate filename
	String filepath = generate_filename("traces");

	// Write to file
	Ref<FileAccess> file;
	if (mode == EXPORT_MODE_APPEND && FileAccess::exists(filepath)) {
		file = FileAccess::open(filepath, FileAccess::READ_WRITE);
		if (file.is_valid()) {
			file->seek_end();
			file->store_string("\n");
		}
	} else {
		file = FileAccess::open(filepath, FileAccess::WRITE);
	}

	if (file.is_null()) {
		ERR_PRINT(vformat("OpenTelemetryExporterFile: Failed to open file for writing: %s", filepath));
		return ERR_FILE_CANT_WRITE;
	}

	file->store_string(json_data);
	file->flush();

	return OK;
}

Error OpenTelemetryExporterFile::export_metrics(Ref<OpenTelemetryState> p_state) {
	if (p_state.is_null()) {
		return ERR_INVALID_PARAMETER;
	}

	Error err = ensure_directory_exists();
	if (err != OK) {
		return err;
	}

	String json_data = document->serialize_metrics(p_state);

	if (pretty_print) {
		JSON json_parser;
		Error parse_err = json_parser.parse(json_data);
		if (parse_err == OK) {
			json_data = JSON::stringify(json_parser.get_data(), "\t", false);
		}
	}

	String filepath = generate_filename("metrics");

	Ref<FileAccess> file;
	if (mode == EXPORT_MODE_APPEND && FileAccess::exists(filepath)) {
		file = FileAccess::open(filepath, FileAccess::READ_WRITE);
		if (file.is_valid()) {
			file->seek_end();
			file->store_string("\n");
		}
	} else {
		file = FileAccess::open(filepath, FileAccess::WRITE);
	}

	if (file.is_null()) {
		ERR_PRINT(vformat("OpenTelemetryExporterFile: Failed to open file for writing: %s", filepath));
		return ERR_FILE_CANT_WRITE;
	}

	file->store_string(json_data);
	file->flush();

	return OK;
}

Error OpenTelemetryExporterFile::export_logs(Ref<OpenTelemetryState> p_state) {
	if (p_state.is_null()) {
		return ERR_INVALID_PARAMETER;
	}

	Error err = ensure_directory_exists();
	if (err != OK) {
		return err;
	}

	String json_data = document->serialize_logs(p_state);

	if (pretty_print) {
		JSON json_parser;
		Error parse_err = json_parser.parse(json_data);
		if (parse_err == OK) {
			json_data = JSON::stringify(json_parser.get_data(), "\t", false);
		}
	}

	String filepath = generate_filename("logs");

	Ref<FileAccess> file;
	if (mode == EXPORT_MODE_APPEND && FileAccess::exists(filepath)) {
		file = FileAccess::open(filepath, FileAccess::READ_WRITE);
		if (file.is_valid()) {
			file->seek_end();
			file->store_string("\n");
		}
	} else {
		file = FileAccess::open(filepath, FileAccess::WRITE);
	}

	if (file.is_null()) {
		ERR_PRINT(vformat("OpenTelemetryExporterFile: Failed to open file for writing: %s", filepath));
		return ERR_FILE_CANT_WRITE;
	}

	file->store_string(json_data);
	file->flush();

	return OK;
}
