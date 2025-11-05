/**************************************************************************/
/*  litrt_compiled_model.h                                                */
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

// Forward declare to avoid conflict with typedefs
// Include our headers (which forward declare types, not LiteRT headers)
#include "litrt_tensor_buffer.h"
#include "litrt_environment.h"
#include "litrt_model.h"

// Don't include litert headers here to avoid typedef conflicts
// The typedef will be handled in the .cpp file
// Forward declare to avoid conflict with typedef LiteRtCompiledModel (which is LiteRtCompiledModelT*)
// Note: Our class name conflicts with the LiteRT typedef, so we use a handle type
class LiteRtCompiledModelT;
typedef class LiteRtCompiledModelT* LiteRtCompiledModelHandle;

class LiteRtCompiledModelRef : public RefCounted {
	GDCLASS(LiteRtCompiledModelRef, RefCounted);

	// Use opaque pointer to avoid name collision with typedef LiteRtCompiledModel
	LiteRtCompiledModelHandle compiled_model = nullptr;
	Ref<LiteRtEnvironmentRef> environment;
	Ref<LiteRtModelRef> model;

protected:
	static void _bind_methods();

public:
	LiteRtCompiledModelRef();
	~LiteRtCompiledModelRef();

	// Create compiled model from environment and model
	Error create(Ref<LiteRtEnvironmentRef> p_environment, Ref<LiteRtModelRef> p_model);

	// Run inference
	Error run(int p_signature_index, const TypedArray<LiteRtTensorBufferRef> &p_inputs, TypedArray<LiteRtTensorBufferRef> p_outputs);

	// Get the underlying handle
	LiteRtCompiledModelHandle get_handle() const { return compiled_model; }

	// Check if compiled model is valid
	bool is_valid() const { return compiled_model != nullptr; }

	// Get number of input buffers required
	int get_num_inputs(int p_signature_index) const;

	// Get number of output buffers required
	int get_num_outputs(int p_signature_index) const;
};
