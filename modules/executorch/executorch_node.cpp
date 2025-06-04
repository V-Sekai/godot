/**************************************************************************/
/*  executorch_node.cpp                                                   */
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

#include "executorch_node.h"
#include "core/object/class_db.h"

ExecuTorchNode::ExecuTorchNode() {
	inference_ = std::make_unique<ExecuTorchInference>();
	auto_load = false;
}

ExecuTorchNode::~ExecuTorchNode() {
	// Unique pointer will automatically clean up
}

void ExecuTorchNode::_bind_methods() {
	// Model management
	ClassDB::bind_method(D_METHOD("load_model", "path"), &ExecuTorchNode::load_model);
	ClassDB::bind_method(D_METHOD("unload_model"), &ExecuTorchNode::unload_model);
	ClassDB::bind_method(D_METHOD("is_model_loaded"), &ExecuTorchNode::is_model_loaded);

	// Inference
	ClassDB::bind_method(D_METHOD("predict", "input"), &ExecuTorchNode::predict);
	ClassDB::bind_method(D_METHOD("predict_named", "inputs"), &ExecuTorchNode::predict_named);

	// Properties
	ClassDB::bind_method(D_METHOD("set_model_path", "path"), &ExecuTorchNode::set_model_path);
	ClassDB::bind_method(D_METHOD("get_model_path"), &ExecuTorchNode::get_model_path);
	ClassDB::bind_method(D_METHOD("set_auto_load", "enable"), &ExecuTorchNode::set_auto_load);
	ClassDB::bind_method(D_METHOD("get_auto_load"), &ExecuTorchNode::get_auto_load);

	// Model info
	ClassDB::bind_method(D_METHOD("get_input_names"), &ExecuTorchNode::get_input_names);
	ClassDB::bind_method(D_METHOD("get_output_names"), &ExecuTorchNode::get_output_names);
	ClassDB::bind_method(D_METHOD("get_input_shape", "name"), &ExecuTorchNode::get_input_shape);
	ClassDB::bind_method(D_METHOD("get_output_shape", "name"), &ExecuTorchNode::get_output_shape);

	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.pte,*.et"), "set_model_path", "get_model_path");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_load"), "set_auto_load", "get_auto_load");

	// Signals
	ADD_SIGNAL(MethodInfo("model_loaded"));
	ADD_SIGNAL(MethodInfo("model_unloaded"));
	ADD_SIGNAL(MethodInfo("inference_completed", PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "result")));
}

void ExecuTorchNode::_ready() {
	if (auto_load && !model_path.is_empty()) {
		load_model(model_path);
	}
}

void ExecuTorchNode::_exit_tree() {
	if (inference_) {
		unload_model();
	}
}

bool ExecuTorchNode::load_model(const String &path) {
	if (!inference_) {
		print_error("ExecuTorch inference not initialized");
		return false;
	}

	std::string std_path = path.utf8().get_data();
	bool success = inference_->load_model(std_path);

	if (success) {
		model_path = path;
		emit_signal("model_loaded");
		print_line("ExecuTorch model loaded: " + path);
	} else {
		print_error("Failed to load ExecuTorch model: " + path);
	}

	return success;
}

void ExecuTorchNode::unload_model() {
	// Currently no unload method in ExecuTorchInference
	// In a real implementation, you'd add this
	model_path = "";
	emit_signal("model_unloaded");
	print_line("ExecuTorch model unloaded");
}

bool ExecuTorchNode::is_model_loaded() const {
	return inference_ && inference_->get_model().is_valid() && inference_->get_model()->is_loaded();
}

PackedFloat32Array ExecuTorchNode::predict(const PackedFloat32Array &input) {
	if (!is_model_loaded()) {
		print_error("No model loaded");
		return PackedFloat32Array();
	}

	// Convert PackedFloat32Array to std::vector<float>
	Vector<float> input_vec;
	input_vec.resize(input.size());
	for (int i = 0; i < input.size(); i++) {
		input_vec.push_back(input[i]);
	}

	// Run inference
	Vector<float> result = inference_->predict(input_vec);

	// Convert back to PackedFloat32Array
	PackedFloat32Array output;
	output.resize(result.size());
	for (size_t i = 0; i < result.size(); i++) {
		output.write[i] = result[i];
	}

	emit_signal("inference_completed", output);
	return output;
}

Dictionary ExecuTorchNode::predict_named(const Dictionary &inputs) {
	if (!is_model_loaded()) {
		print_error("No model loaded");
		return Dictionary();
	}

	// TODO: Implement named input/output prediction
	print_line("Named prediction not yet implemented");
	return Dictionary();
}

void ExecuTorchNode::set_model_path(const String &path) {
	model_path = path;
}

String ExecuTorchNode::get_model_path() const {
	return model_path;
}

void ExecuTorchNode::set_auto_load(bool enable) {
	auto_load = enable;
}

bool ExecuTorchNode::get_auto_load() const {
	return auto_load;
}

PackedStringArray ExecuTorchNode::get_input_names() const {
	if (!is_model_loaded()) {
		return PackedStringArray();
	}

	// TODO: Convert std::vector<std::string> to PackedStringArray
	print_line("get_input_names not yet implemented");
	return PackedStringArray();
}

PackedStringArray ExecuTorchNode::get_output_names() const {
	if (!is_model_loaded()) {
		return PackedStringArray();
	}

	// TODO: Convert std::vector<std::string> to PackedStringArray
	print_line("get_output_names not yet implemented");
	return PackedStringArray();
}

PackedInt64Array ExecuTorchNode::get_input_shape(const String &name) const {
	if (!is_model_loaded()) {
		return PackedInt64Array();
	}

	// TODO: Convert std::vector<int64_t> to PackedInt64Array
	print_line("get_input_shape not yet implemented");
	return PackedInt64Array();
}

PackedInt64Array ExecuTorchNode::get_output_shape(const String &name) const {
	if (!is_model_loaded()) {
		return PackedInt64Array();
	}

	// TODO: Convert std::vector<int64_t> to PackedInt64Array
	print_line("get_output_shape not yet implemented");
	return PackedInt64Array();
}
