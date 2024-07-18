/**************************************************************************/
/*  resource_loader_wasm.cpp                                              */
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

/* resource_loader_wasm.cpp */

#include "resource_loader_wasm.h"

#include "modules/wasgo/src/wasgo_runtime.h"
#include "resource_wasm.h"

Ref<Resource> ResourceFormatLoaderWasm::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<WasGoRuntime> runtime = memnew(WasGoRuntime);
	Ref<WasmResource> wasm = memnew(WasmResource);
	if (r_error) {
		*r_error = OK;
	}
	*r_error = wasm->load_file(p_path, runtime);
	return runtime;
}

void ResourceFormatLoaderWasm::get_recognized_extensions(List<String> *r_extensions) const {
	if (!r_extensions->find("wasm")) {
		r_extensions->push_back("wasm");
	}
}

String ResourceFormatLoaderWasm::get_resource_type(const String &p_path) const {
	return "WasGoRuntime";
}

bool ResourceFormatLoaderWasm::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Resource");
}
