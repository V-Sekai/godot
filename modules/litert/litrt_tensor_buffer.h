/**************************************************************************/
/*  litrt_tensor_buffer.h                                                 */
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

#pragma once

#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"

// Forward declare to avoid conflict with typedef LiteRtTensorBuffer (which is LiteRtTensorBufferT*)
// Don't include litert_tensor_buffer.h here to avoid typedef conflict
// The typedef will be handled in the .cpp file
class LiteRtTensorBufferT;
typedef class LiteRtTensorBufferT* LiteRtTensorBufferHandle;

class LiteRtTensorBufferRef : public RefCounted {
	GDCLASS(LiteRtTensorBufferRef, RefCounted);

	// Use opaque pointer to avoid name collision with typedef LiteRtTensorBuffer
	LiteRtTensorBufferHandle tensor_buffer = nullptr;
	PackedFloat32Array data_array;
	void *host_memory = nullptr;

protected:
	static void _bind_methods();

public:
	LiteRtTensorBufferRef();
	~LiteRtTensorBufferRef();

	// Create tensor buffer from PackedFloat32Array
	Error create_from_array(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape);

	// Get data as PackedFloat32Array
	PackedFloat32Array get_data() const;

	// Get the underlying handle
	LiteRtTensorBufferHandle get_handle() const { return tensor_buffer; }

	// Check if tensor buffer is valid
	bool is_valid() const { return tensor_buffer != nullptr; }

	// Get shape
	PackedInt32Array get_shape() const;
};
