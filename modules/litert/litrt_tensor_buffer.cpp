/**************************************************************************/
/*  litrt_tensor_buffer.cpp                                               */
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

#include "litrt_tensor_buffer.h"

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"

void LiteRtTensorBuffer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_from_array", "data", "shape"), &LiteRtTensorBuffer::create_from_array);
	ClassDB::bind_method(D_METHOD("get_data"), &LiteRtTensorBuffer::get_data);
	ClassDB::bind_method(D_METHOD("is_valid"), &LiteRtTensorBuffer::is_valid);
	ClassDB::bind_method(D_METHOD("get_shape"), &LiteRtTensorBuffer::get_shape);
}

LiteRtTensorBuffer::LiteRtTensorBuffer() {
}

LiteRtTensorBuffer::~LiteRtTensorBuffer() {
	if (tensor_buffer != nullptr) {
		LiteRtDestroyTensorBuffer(tensor_buffer);
		tensor_buffer = nullptr;
	}
	if (host_memory != nullptr) {
		memfree(host_memory);
		host_memory = nullptr;
	}
}

Error LiteRtTensorBuffer::create_from_array(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape) {
	if (tensor_buffer != nullptr) {
		return ERR_ALREADY_EXISTS;
	}

	if (p_data.size() == 0 || p_shape.size() == 0) {
		return ERR_INVALID_PARAMETER;
	}

	// Calculate total size
	int total_size = 1;
	for (int i = 0; i < p_shape.size(); i++) {
		total_size *= p_shape[i];
	}

	if (total_size != p_data.size()) {
		return ERR_INVALID_PARAMETER;
	}

	// Create ranked tensor type
	LiteRtRankedTensorType tensor_type;
	tensor_type.type_id = kLiteRtTensorTypeIdFloat32;
	tensor_type.rank = p_shape.size();
	for (int i = 0; i < p_shape.size(); i++) {
		tensor_type.shape[i] = p_shape[i];
	}

	// Allocate aligned memory
	size_t buffer_size = p_data.size() * sizeof(float);
	host_memory = memalloc(buffer_size);
	if (host_memory == nullptr) {
		return ERR_OUT_OF_MEMORY;
	}

	// Copy data
	memcpy(host_memory, p_data.ptr(), buffer_size);
	data_array = p_data;

	// Create tensor buffer
	LiteRtStatus status = LiteRtCreateTensorBufferFromHostMemory(
		&tensor_type,
		host_memory,
		buffer_size,
		nullptr, // deallocator - we'll manage memory ourselves
		&tensor_buffer
	);

	if (status != kLiteRtStatusOk) {
		memfree(host_memory);
		host_memory = nullptr;
		return FAILED;
	}

	return OK;
}

PackedFloat32Array LiteRtTensorBuffer::get_data() const {
	PackedFloat32Array result;

	if (tensor_buffer == nullptr) {
		return result;
	}

	void *host_mem = nullptr;
	LiteRtStatus status = LiteRtGetTensorBufferHostMemory(tensor_buffer, &host_mem);
	if (status != kLiteRtStatusOk || host_mem == nullptr) {
		return result;
	}

	// Get tensor type to determine size
	// For now, return the stored data_array
	return data_array;
}

PackedInt32Array LiteRtTensorBuffer::get_shape() const {
	PackedInt32Array result;

	if (tensor_buffer == nullptr) {
		return result;
	}

	// TODO: Get shape from tensor buffer
	// For now return empty
	return result;
}

