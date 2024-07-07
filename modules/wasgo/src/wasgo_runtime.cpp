/**************************************************************************/
/*  wasgo_runtime.cpp                                                     */
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

#include "wasgo_runtime.h"
#include "include/wasgo_function_table.h"
#include "wasm_export.h"

char WasGoRuntime::global_heap_buf[512 * 1024 * 100];
WasGoRuntime *WasGoRuntime::singleton = NULL;
RuntimeInitArgs WasGoRuntime::init_args;

WasGoRuntime *WasGoRuntime::get_singleton() {
	return singleton;
}

// wasm_module_t WasGoRuntime::load_module(String script_path){
RID WasGoRuntime::load_module(Vector<uint8_t> wasm_code, String &r_error) {
	MutexLock l(mutex);
	int buf_size = wasm_code.size(); // buffer_length
	uint8_t *buffer = (uint8_t *)wasm_code.ptr(); // buffer

	char error_buf[128];
	error_buf[0] = 0;

	wasm_module_t res = wasm_runtime_load(buffer, buf_size, error_buf, sizeof(error_buf));
	r_error = error_buf;
	if (!res) {
		return RID();
	}
	return module_rids.make_rid(res);
}

void WasGoRuntime::unload_module(RID module_rid) {
	wasm_module_t module = module_rids.get_or_null(module_rid);
	if (module) {
		wasm_runtime_unload(module);
		module_rids.free(module_rid);
	}
}

wasm_module_inst_t WasGoRuntime::instantiate_module(RID module_rid,
		uint32_t stack_size, uint32_t heap_size, String &r_error_str) {
	if (module_rid == RID()) {
		r_error_str = "Null RID passed to instantiate_module.";
		return nullptr;
	}
	char error_buf[128];
	error_buf[0] = 0;
	wasm_module_t module = module_rids.get_or_null(module_rid);
	if (module == nullptr) {
		r_error_str = "Invalid RID passed to instantiate_module.";
		return nullptr;
	}

	wasm_module_inst_t module_inst = wasm_runtime_instantiate(module,
			stack_size,
			heap_size,
			error_buf,
			sizeof(error_buf));
	if (module_inst == nullptr) {
		r_error_str = error_buf;
	}
	return module_inst;
}

WasGoRuntime::WasGoRuntime() {
	singleton = this;
	singleton->initialize(native_symbols, sizeof(native_symbols) / sizeof(NativeSymbol));
}

WasGoRuntime::~WasGoRuntime() {
	// void get_key_list(List<TKey> *p_keys) const {

	List<RID> keys;
	module_rids.get_owned_list(&keys);
	List<RID>::Element *curr = keys.front();
	while (curr && curr->next() != nullptr) {
		WASMModuleCommon *module_rid = module_rids.get_or_null(curr->get());
		if (!module_rid) {
			curr = curr->next();
			continue;
		}
		wasm_runtime_unload(module_rid);
		curr = curr->next();
	}
	wasm_runtime_destroy();
}

void WasGoRuntime::initialize(NativeSymbol symbols[], int native_symbol_size) {
	memset(&init_args, 0, sizeof(RuntimeInitArgs));

	init_args.mem_alloc_type = Alloc_With_Pool;
	init_args.mem_alloc_option.pool.heap_buf = WasGoRuntime::global_heap_buf;
	init_args.mem_alloc_option.pool.heap_size = sizeof(WasGoRuntime::global_heap_buf);

	// Native symbols need below registration phase
	init_args.n_native_symbols = native_symbol_size;
	init_args.native_module_name = "env";
	init_args.native_symbols = symbols;

	if (!wasm_runtime_full_init(&init_args)) {
		print_error("Wasgo runtime environment cannot be ran.");
	} else {
		print_verbose("Wasgo runtime environment running.");
	}
}