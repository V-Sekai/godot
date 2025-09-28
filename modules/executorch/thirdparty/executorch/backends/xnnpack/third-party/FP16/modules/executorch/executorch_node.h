/**************************************************************************/
/*  executorch_node.h                                                     */
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

#include "core/string/ustring.h"
#include "executorch_inference.h"
#include "scene/main/node.h"
#include <memory>

class ExecuTorchNode : public Node {
	GDCLASS(ExecuTorchNode, Node);

private:
	std::unique_ptr<ExecuTorchInference> inference_;
	String model_path;
	bool auto_load;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	ExecuTorchNode();
	~ExecuTorchNode();

	// Model management
	bool load_model(const String &path);
	void unload_model();
	bool is_model_loaded() const;

	// Inference
	virtual PackedFloat32Array predict(const PackedFloat32Array &input);
	Dictionary predict_named(const Dictionary &inputs);

	// Properties
	void set_model_path(const String &path);
	String get_model_path() const;
	void set_auto_load(bool enable);
	bool get_auto_load() const;

	// Model info
	PackedStringArray get_input_names() const;
	PackedStringArray get_output_names() const;
	PackedInt64Array get_input_shape(const String &name) const;
	PackedInt64Array get_output_shape(const String &name) const;
};
