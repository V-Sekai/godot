/**************************************************************************/
/*  litrt_compiled_model.cpp                                              */
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

// Forward declare LiteRT types to avoid conflicts
// We'll access the typedefs via function pointers and explicit casts
class LiteRtCompiledModelT;
class LiteRtTensorBufferT;
class LiteRtEnvironmentT;
class LiteRtModelT;

// Include our class header first (defines our wrapper classes)
#include "litrt_compiled_model.h"

// Now include LiteRT headers to get typedefs for API calls
// The typedefs will shadow our class names, so we use function pointers and casts
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"

#include "core/error/error_macros.h"

// After including LiteRT headers, the typedefs are defined:
// - LiteRtCompiledModel is typedef LiteRtCompiledModelT*
// - LiteRtTensorBuffer is typedef LiteRtTensorBufferT*
// - LiteRtEnvironment is typedef LiteRtEnvironmentT*
// - LiteRtModel is typedef LiteRtModelT*
//
// Our wrapper classes use handle types (LiteRt...Handle) to avoid conflicts.
// When calling LiteRT API functions, we cast our handles to the typedef types.
//
// Helper typedefs to reference the LiteRT API types without name conflicts
// LiteRT typedefs are: typedef class NameT* Name
// We create our own typedefs that directly reference the underlying pointer types
// This avoids the name conflict with our wrapper classes
typedef class LiteRtCompiledModelT* LiteRtCompiledModelApi;
typedef class LiteRtTensorBufferT* LiteRtTensorBufferApi;
typedef class LiteRtEnvironmentT* LiteRtEnvironmentApi;
typedef class LiteRtModelT* LiteRtModelApi;

void LiteRtCompiledModelRef::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create", "environment", "model"), &LiteRtCompiledModelRef::create);
	ClassDB::bind_method(D_METHOD("run", "signature_index", "inputs", "outputs"), &LiteRtCompiledModelRef::run);
	ClassDB::bind_method(D_METHOD("is_valid"), &LiteRtCompiledModelRef::is_valid);
	ClassDB::bind_method(D_METHOD("get_num_inputs", "signature_index"), &LiteRtCompiledModelRef::get_num_inputs);
	ClassDB::bind_method(D_METHOD("get_num_outputs", "signature_index"), &LiteRtCompiledModelRef::get_num_outputs);
}

LiteRtCompiledModelRef::LiteRtCompiledModelRef() {
}

LiteRtCompiledModelRef::~LiteRtCompiledModelRef() {
	if (compiled_model != nullptr) {
		// Cast our handle to LiteRT API type (both are LiteRtCompiledModelT*)
	LiteRtCompiledModelApi handle = (LiteRtCompiledModelApi)(compiled_model);
	// Call via function pointer using underlying pointer type to avoid name conflict
	typedef void (*DestroyFunc)(LiteRtCompiledModelT*);
	DestroyFunc destroy_func = (DestroyFunc)LiteRtDestroyCompiledModel;
	destroy_func((LiteRtCompiledModelT*)(handle));
		compiled_model = nullptr;
	}
}

Error LiteRtCompiledModelRef::create(Ref<LiteRtEnvironmentRef> p_environment, Ref<LiteRtModelRef> p_model) {
	if (compiled_model != nullptr) {
		return ERR_ALREADY_EXISTS;
	}

	if (p_environment.is_null() || !p_environment->is_valid()) {
		return ERR_INVALID_PARAMETER;
	}

	if (p_model.is_null() || !p_model->is_valid()) {
		return ERR_INVALID_PARAMETER;
	}

	// Cast our handles to LiteRT API types (all are pointers to the T structs)
	LiteRtEnvironmentApi env_handle = (LiteRtEnvironmentApi)(p_environment->get_handle());
	LiteRtModelApi model_handle = (LiteRtModelApi)(p_model->get_handle());
	
	// Use function pointer with underlying pointer types to avoid name conflicts
	// The actual LiteRT function signature uses typedefs, but we define our function pointer
	// using the underlying types to avoid conflicts with our class names
	typedef LiteRtStatus (*CreateCompiledModelFunc)(LiteRtEnvironmentT*, LiteRtModelT*, void*, LiteRtCompiledModelT**);
	CreateCompiledModelFunc create_func = (CreateCompiledModelFunc)LiteRtCreateCompiledModel;
	
	LiteRtCompiledModelT* compiled_model_ptr = nullptr;
	LiteRtStatus status = create_func(
			(LiteRtEnvironmentT*)(env_handle),
			(LiteRtModelT*)(model_handle),
			nullptr, // compilation_options - null for now
			&compiled_model_ptr);
	compiled_model = (LiteRtCompiledModelHandle)(compiled_model_ptr); // Both are LiteRtCompiledModelT*, same type

	if (status != kLiteRtStatusOk) {
		compiled_model = nullptr;
		return FAILED;
	}

	environment = p_environment;
	model = p_model;

	return OK;
}

Error LiteRtCompiledModelRef::run(int p_signature_index, const TypedArray<LiteRtTensorBufferRef> &p_inputs, TypedArray<LiteRtTensorBufferRef> p_outputs) {
	if (compiled_model == nullptr) {
		return ERR_UNCONFIGURED;
	}

	// Convert TypedArray to C arrays
	size_t num_inputs = p_inputs.size();
	size_t num_outputs = p_outputs.size();

	if (num_inputs == 0 || num_outputs == 0) {
		return ERR_INVALID_PARAMETER;
	}

	// Allocate arrays of LiteRT API pointers (LiteRtTensorBuffer = LiteRtTensorBufferT*)
	LiteRtTensorBufferApi *input_buffers = (LiteRtTensorBufferApi *)memalloc(sizeof(LiteRtTensorBufferApi) * num_inputs);
	LiteRtTensorBufferApi *output_buffers = (LiteRtTensorBufferApi *)memalloc(sizeof(LiteRtTensorBufferApi) * num_outputs);

	for (size_t i = 0; i < num_inputs; i++) {
		Ref<LiteRtTensorBufferRef> buf = p_inputs[i];
		if (buf.is_null() || !buf->is_valid()) {
			memfree(input_buffers);
			memfree(output_buffers);
			return ERR_INVALID_PARAMETER;
		}
		// Cast from our handle type to LiteRT API type (both are LiteRtTensorBufferT*)
		input_buffers[i] = (LiteRtTensorBufferApi)(buf->get_handle());
	}

	for (size_t i = 0; i < num_outputs; i++) {
		Ref<LiteRtTensorBufferRef> buf = p_outputs[i];
		if (buf.is_null() || !buf->is_valid()) {
			memfree(input_buffers);
			memfree(output_buffers);
			return ERR_INVALID_PARAMETER;
		}
		// Cast from our handle type to LiteRT API type (both are LiteRtTensorBufferT*)
		output_buffers[i] = (LiteRtTensorBufferApi)(buf->get_handle());
	}

	// Cast our handle to LiteRT API type (both are LiteRtCompiledModelT*)
	// Use function pointer with underlying pointer types to avoid name conflict
	typedef LiteRtStatus (*RunCompiledModelFunc)(
			LiteRtCompiledModelT*,
			LiteRtParamIndex,
			size_t,
			LiteRtTensorBufferT**,
			size_t,
			LiteRtTensorBufferT**);
	RunCompiledModelFunc run_func = (RunCompiledModelFunc)LiteRtRunCompiledModel;
	LiteRtCompiledModelApi handle = (LiteRtCompiledModelApi)(compiled_model);
	LiteRtStatus status = run_func(
			(LiteRtCompiledModelT*)(handle),
			static_cast<LiteRtParamIndex>(p_signature_index),
			num_inputs,
			(LiteRtTensorBufferT**)(input_buffers),
			num_outputs,
			(LiteRtTensorBufferT**)(output_buffers));

	memfree(input_buffers);
	memfree(output_buffers);

	if (status != kLiteRtStatusOk) {
		return FAILED;
	}

	return OK;
}

int LiteRtCompiledModelRef::get_num_inputs(int p_signature_index) const {
	if (compiled_model == nullptr) {
		return 0;
	}

	// TODO: Implement get_num_inputs - API might not be available or have different name
	// For now, return 0 as placeholder
	return 0;
}

int LiteRtCompiledModelRef::get_num_outputs(int p_signature_index) const {
	if (compiled_model == nullptr) {
		return 0;
	}

	// TODO: Implement get_num_outputs - API might not be available or have different name
	// For now, return 0 as placeholder
	return 0;
}
