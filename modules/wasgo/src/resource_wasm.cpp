/**************************************************************************/
/*  resource_wasm.cpp                                                     */
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

/* resource_wasm.cpp */

#include "resource_wasm.h"

#include "core/core_bind.h"
#include "core/io/file_access.h"

#include "wasgo_runtime.h"

Error WasmResource::load_file(const String &p_path, Ref<WasGoRuntime> p_wasgo_runtime) {
	if (!wasm_buf.is_empty()) {
		return Error::ERR_ALREADY_IN_USE;
	}
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V_MSG(err != OK, err, vformat("We couldn't read the file: %s\n", p_path));
	ERR_FAIL_COND_V_MSG(file.is_null(), FAILED, vformat("We couldn't read the file: %s\n", p_path));
	Vector<uint8_t> buf;
	buf.resize(file->get_length());
	uint8_t *w = buf.ptrw();
	uint32_t read_amt = file->get_buffer(&w[0], file->get_length());
	if (read_amt != buf.size()) {
		return file->get_error();
	}
	set_wasm_buffer(buf, p_wasgo_runtime);
	if (module_rid == RID()) {
		return Error::ERR_COMPILATION_FAILED;
	}
	return OK;
}

void WasmResource::set_wasm_buffer(Vector<uint8_t> p_buffer, Ref<WasGoRuntime> p_runtime) {
	if (module_rid != RID()) {
		return; // Avoid mutating already-loaded module.
	}
	ERR_FAIL_COND(!p_buffer.size());
	String err_string;
	wasm_buf = p_buffer;
	module_rid = p_runtime->load_module(wasm_buf, err_string);

	ERR_FAIL_COND_MSG(module_rid == RID(), vformat("Failed to load wasm module %s: \"%s\".", get_path(), err_string));
}

RID WasmResource::get_rid() const {
	return module_rid;
}

void WasmResource::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_wasm_buffer", "buffer", "runtime"), &WasmResource::set_wasm_buffer);
	ClassDB::bind_method(D_METHOD("get_wasm_buffer"), &WasmResource::get_wasm_buffer);
}
