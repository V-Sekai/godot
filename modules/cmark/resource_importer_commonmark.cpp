/**************************************************************************/
/*  resource_importer_commonmark.cpp                                      */
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

#include "resource_importer_commonmark.h"
#include "core/error/error_macros.h"
#include "core/io/file_access_pack.h"

#include "cmark-gfm.h"
#include "core/os/memory.h"
#include <stdint.h>

String ResourceImporterCommonmark::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterCommonmark::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
}

bool ResourceImporterCommonmark::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

String ResourceImporterCommonmark::get_importer_name() const {
	return "Commonmark";
}

String ResourceImporterCommonmark::get_visible_name() const {
	return "Commonmark";
}

void ResourceImporterCommonmark::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("md");
}

String ResourceImporterCommonmark::get_save_extension() const {
	return "res";
}

String ResourceImporterCommonmark::get_resource_type() const {
	return "CommonmarkData";
}

int ResourceImporterCommonmark::get_preset_count() const {
	return 0;
}

Error ResourceImporterCommonmark::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	Ref<FileAccess> file = FileAccess::create(FileAccess::ACCESS_RESOURCES);
	Error err;
	String commonmark = file->get_file_as_string(p_source_file, &err);
	ERR_FAIL_COND_V(err != OK, FAILED);
	Ref<CommonmarkData> commonmark_data;
	commonmark_data.instantiate();
	commonmark_data->set_html(commonmark_data->convert_commonmark(commonmark));
	return ResourceSaver::save(commonmark_data, vformat("%s%s", p_save_path, ".res"));
}

String CommonmarkData::get_html() const {
	return html;
}

void CommonmarkData::set_html(const String p_string) {
	html = p_string;
}

String CommonmarkData::convert_commonmark(String p_string) {
	Vector<uint8_t> string_bytes = p_string.to_utf8_buffer();
	char *cmark_bytes = cmark_markdown_to_html((const char *)string_bytes.ptr(), string_bytes.size() - 1, CMARK_OPT_DEFAULT);
	String new_string;
	new_string.parse_utf8(cmark_bytes);
	if (cmark_bytes) {
		free(cmark_bytes);
	}
	return new_string;
}
