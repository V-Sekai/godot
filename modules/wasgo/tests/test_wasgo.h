/**************************************************************************/
/*  test_wasgo.h                                                          */
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

#ifndef TEST_WASGO_H
#define TEST_WASGO_H

#include "modules/wasgo/src/wasgo_runtime.h"
#include "tests/test_macros.h"

#include "modules/wasgo/src/resource_wasm.h"
#include "modules/wasgo/src/wasgo_state.h"
#include "modules/wasgo/thirdparty/wasm-micro-runtime/core/iwasm/include/wasm_export.h"
#include "turning_blob.h"
#include <stdlib.h>

namespace TestWasgo {

TEST_CASE("[Modules][WasGo] Turning Callable") {
	RuntimeInitArgs init_args;
	memset(&init_args, 0, sizeof(RuntimeInitArgs));

	Vector<uint8_t> wasm_buffer;
    const size_t TURNING_WASM_len = sizeof(TURNING_WASM) / sizeof(TURNING_WASM[0]);
	wasm_buffer.resize(TURNING_WASM_len);
	memcpy(wasm_buffer.ptrw(), TURNING_WASM, TURNING_WASM_len);

	Ref<WasGoRuntime> runtime = memnew(WasGoRuntime);
	Ref<WasmResource> reference = memnew(WasmResource);
	reference->set_wasm_buffer(wasm_buffer, runtime);
	REQUIRE_MESSAGE(reference.is_valid(), "Creating WasGoState.");

	WasGoState *state = memnew(WasGoState);
	state->set_wasm_script(reference, runtime);
	memdelete(state);
	// MESSAGE("Creating Wasm callable.");
	// WasGoCallable callable = WasGoCallable(state, "test");
	// Variant var = Variant(420);
	// const Variant *args = &var;
	// Variant return_val = Variant(0);
	// Callable::CallError r_call_error;
	// callable.call(&args, 0, return_val, r_call_error);
}
} // namespace TestWasgo

#endif // TEST_WASGO_H
