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

// Include the LiteRT headers here to get the typedefs
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_layout.h"

void LiteRtTensorBufferRef::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_from_array", "data", "shape"), &LiteRtTensorBufferRef::create_from_array);
	ClassDB::bind_method(D_METHOD("get_data"), &LiteRtTensorBufferRef::get_data);
	ClassDB::bind_method(D_METHOD("is_valid"), &LiteRtTensorBufferRef::is_valid);
	ClassDB::bind_method(D_METHOD("get_shape"), &LiteRtTensorBufferRef::get_shape);
}

LiteRtTensorBufferRef::LiteRtTensorBufferRef() {
}

LiteRtTensorBufferRef::~LiteRtTensorBufferRef() {
	if (tensor_buffer != nullptr) {
		// Cast handle to typedef type for LiteRT API (both are pointers)
		LiteRtTensorBuffer handle = reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer);
		LiteRtDestroyTensorBuffer(handle);
		tensor_buffer = nullptr;
	}
	if (host_memory != nullptr) {
		memfree(host_memory);
		host_memory = nullptr;
	}
}

Error LiteRtTensorBufferRef::create_from_array(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape) {
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
	// Note: LiteRtRankedTensorType only has element_type and layout
	// The layout encodes the shape information
	LiteRtRankedTensorType tensor_type;
	tensor_type.element_type = kLiteRtElementTypeFloat32;
	// Build layout from shape
	LiteRtLayout layout;
	layout.rank = p_shape.size();
	layout.has_strides = false;
	for (int i = 0; i < p_shape.size() && i < LITERT_TENSOR_MAX_RANK; i++) {
		layout.dimensions[i] = p_shape[i];
	}
	// Zero out remaining dimensions
	for (int i = p_shape.size(); i < LITERT_TENSOR_MAX_RANK; i++) {
		layout.dimensions[i] = 0;
	}
	tensor_type.layout = layout;

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
	// Use the typedef type from litert headers for API call
	LiteRtTensorBuffer buffer = nullptr;
	LiteRtStatus status = LiteRtCreateTensorBufferFromHostMemory(
			&tensor_type,
			host_memory,
			buffer_size,
			nullptr, // deallocator - we'll manage memory ourselves
			&buffer);
	tensor_buffer = reinterpret_cast<LiteRtTensorBufferHandle>(buffer); // Assign to our handle type

	if (status != kLiteRtStatusOk) {
		memfree(host_memory);
		host_memory = nullptr;
		return FAILED;
	}

	return OK;
}

PackedFloat32Array LiteRtTensorBufferRef::get_data() const {
	PackedFloat32Array result;

	if (tensor_buffer == nullptr) {
		return result;
	}

	void *host_mem = nullptr;
	// Cast handle to typedef type for LiteRT API (both are pointers)
	LiteRtTensorBuffer handle = reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer);
	LiteRtStatus status = LiteRtGetTensorBufferHostMemory(handle, &host_mem);
	if (status != kLiteRtStatusOk || host_mem == nullptr) {
		return result;
	}

	// Get tensor type to determine size
	// For now, return the stored data_array
	return data_array;
}

PackedInt32Array LiteRtTensorBufferRef::get_shape() const {
	PackedInt32Array result;

	if (tensor_buffer == nullptr) {
		return result;
	}

	// TODO: Get shape from tensor buffer
	// For now return empty
	return result;
}
