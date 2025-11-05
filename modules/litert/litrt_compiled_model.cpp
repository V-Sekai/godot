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

#include "litrt_compiled_model.h"

#include "core/error/error_macros.h"

void LiteRtCompiledModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create", "environment", "model"), &LiteRtCompiledModel::create);
	ClassDB::bind_method(D_METHOD("run", "signature_index", "inputs", "outputs"), &LiteRtCompiledModel::run);
	ClassDB::bind_method(D_METHOD("is_valid"), &LiteRtCompiledModel::is_valid);
	ClassDB::bind_method(D_METHOD("get_num_inputs", "signature_index"), &LiteRtCompiledModel::get_num_inputs);
	ClassDB::bind_method(D_METHOD("get_num_outputs", "signature_index"), &LiteRtCompiledModel::get_num_outputs);
}

LiteRtCompiledModel::LiteRtCompiledModel() {
}

LiteRtCompiledModel::~LiteRtCompiledModel() {
	if (compiled_model != nullptr) {
		LiteRtDestroyCompiledModel(compiled_model);
		compiled_model = nullptr;
	}
}

Error LiteRtCompiledModel::create(Ref<LiteRtEnvironment> p_environment, Ref<LiteRtModel> p_model) {
	if (compiled_model != nullptr) {
		return ERR_ALREADY_EXISTS;
	}

	if (p_environment.is_null() || !p_environment->is_valid()) {
		return ERR_INVALID_PARAMETER;
	}

	if (p_model.is_null() || !p_model->is_valid()) {
		return ERR_INVALID_PARAMETER;
	}

	LiteRtStatus status = LiteRtCreateCompiledModel(
			p_environment->get_handle(),
			p_model->get_handle(),
			nullptr, // compilation_options - null for now
			&compiled_model);

	if (status != kLiteRtStatusOk) {
		compiled_model = nullptr;
		return FAILED;
	}

	environment = p_environment;
	model = p_model;

	return OK;
}

Error LiteRtCompiledModel::run(int p_signature_index, const TypedArray<LiteRtTensorBuffer> &p_inputs, TypedArray<LiteRtTensorBuffer> p_outputs) {
	if (compiled_model == nullptr) {
		return ERR_UNCONFIGURED;
	}

	// Convert TypedArray to C arrays
	size_t num_inputs = p_inputs.size();
	size_t num_outputs = p_outputs.size();

	if (num_inputs == 0 || num_outputs == 0) {
		return ERR_INVALID_PARAMETER;
	}

	LiteRtTensorBuffer *input_buffers = (LiteRtTensorBuffer *)memalloc(sizeof(LiteRtTensorBuffer) * num_inputs);
	LiteRtTensorBuffer *output_buffers = (LiteRtTensorBuffer *)memalloc(sizeof(LiteRtTensorBuffer) * num_outputs);

	for (size_t i = 0; i < num_inputs; i++) {
		Ref<LiteRtTensorBuffer> buf = p_inputs[i];
		if (buf.is_null() || !buf->is_valid()) {
			memfree(input_buffers);
			memfree(output_buffers);
			return ERR_INVALID_PARAMETER;
		}
		input_buffers[i] = buf->get_handle();
	}

	for (size_t i = 0; i < num_outputs; i++) {
		Ref<LiteRtTensorBuffer> buf = p_outputs[i];
		if (buf.is_null() || !buf->is_valid()) {
			memfree(input_buffers);
			memfree(output_buffers);
			return ERR_INVALID_PARAMETER;
		}
		output_buffers[i] = buf->get_handle();
	}

	LiteRtStatus status = LiteRtRunCompiledModel(
			compiled_model,
			static_cast<LiteRtParamIndex>(p_signature_index),
			num_inputs,
			input_buffers,
			num_outputs,
			output_buffers);

	memfree(input_buffers);
	memfree(output_buffers);

	if (status != kLiteRtStatusOk) {
		return FAILED;
	}

	return OK;
}

int LiteRtCompiledModel::get_num_inputs(int p_signature_index) const {
	if (compiled_model == nullptr) {
		return 0;
	}

	// This would require getting buffer requirements - simplified for now
	return 0;
}

int LiteRtCompiledModel::get_num_outputs(int p_signature_index) const {
	if (compiled_model == nullptr) {
		return 0;
	}

	// This would require getting buffer requirements - simplified for now
	return 0;
}
