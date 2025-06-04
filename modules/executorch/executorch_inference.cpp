/**************************************************************************/
/*  executorch_inference.cpp                                              */
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

#include "executorch_inference.h"
#include "executorch_runtime.h"

ExecuTorchInference::ExecuTorchInference(bool auto_manage) :
		auto_manage_runtime_(auto_manage) {
	if (auto_manage_runtime_) {
		runtime_ = std::make_unique<ExecuTorchRuntime>();
	}
	model_ = Ref<ExecuTorchResource>();
}

ExecuTorchInference::~ExecuTorchInference() {
}

bool ExecuTorchInference::load_model(const std::string &file_path) {
	if (auto_manage_runtime_ && runtime_) {
		if (!runtime_->initialize()) {
			print_error("Failed to initialize ExecuTorch runtime");
			return false;
		}
	}

	model_ = Ref<ExecuTorchResource>(memnew(ExecuTorchResource));
	bool success = model_->load_from_file(String(file_path.c_str()));
	if (!success) {
		print_error("Failed to load model from: " + String(file_path.c_str()));
		model_ = Ref<ExecuTorchResource>();
		return false;
	}

	print_line("Successfully loaded model: " + String(file_path.c_str()));
	return true;
}

PackedFloat32Array ExecuTorchInference::predict(const PackedFloat32Array &input) {
	if (!model_.is_valid() || !model_->is_loaded()) {
		print_error("Model not loaded");
		return PackedFloat32Array();
	}

	// Convert PackedFloat32Array to std::vector<float>
	std::vector<float> input_vec;
	input_vec.reserve(input.size());
	for (int64_t i = 0; i < input.size(); i++) {
		input_vec.push_back(input[i]);
	}

	// Use the resource's forward method (check what methods are available)
	Dictionary inputs;
	inputs["input"] = input; // Pass the original PackedFloat32Array
	Dictionary outputs = model_->forward(inputs);

	// Extract the output (assuming single output named "output")
	if (outputs.has("output")) {
		return outputs["output"];
	}

	return PackedFloat32Array();
}

void ExecuTorchInference::set_runtime(ExecuTorchRuntime *external_runtime) {
	if (auto_manage_runtime_) {
		// Release our managed runtime
		runtime_.reset();
		auto_manage_runtime_ = false;
	}

	// Note: We're not storing the external runtime pointer here
	// In a real implementation, you'd need to modify ExecuTorchResource
	// to accept and use the external runtime
	print_line("External runtime set (implementation pending)");
}
