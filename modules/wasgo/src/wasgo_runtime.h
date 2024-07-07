/**************************************************************************/
/*  wasgo_runtime.h                                                       */
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

#ifndef WASGO_RUNTIME_H
#define WASGO_RUNTIME_H

#include "core/config/engine.h"
#include "core/io/file_access.h"
#include "core/templates/rid_owner.h"
#include "core/variant/variant.h"
#include "resource_loader_wasm.h"
#include "resource_wasm.h"

struct WASMModuleCommon;
typedef struct WASMModuleCommon *wasm_module_t;
struct WASMModuleInstanceCommon;
typedef struct WASMModuleInstanceCommon *wasm_module_inst_t;

class WasGoRuntime : public Object {
	GDCLASS(WasGoRuntime, Object);
	static WasGoRuntime *singleton;

	Mutex mutex;
	RID_PtrOwner<WASMModuleCommon> module_rids;

public:
	static WasGoRuntime *get_singleton();
	static char global_heap_buf[512 * 1024 * 100];
	static RuntimeInitArgs init_args;
	RID load_module(Vector<uint8_t> wasm_code, String &r_error);
	void unload_module(RID module_rid);

	wasm_module_inst_t instantiate_module(RID module_rid, uint32_t stack_size, uint32_t heap_size, String &r_error);

	WasGoRuntime();
	~WasGoRuntime();
	void initialize(NativeSymbol symbols[], int native_symbol_size);
};

#endif // WASGO_RUNTIME_H
