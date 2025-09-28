/**************************************************************************/
/*  executorch_mv2_demo.cpp                                              */
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

#include "executorch_mv2_demo.h"
#include "core/variant/typed_array.h"

void ExecuTorchMV2Demo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_model_path", "path"), &ExecuTorchMV2Demo::set_model_path);
	ClassDB::bind_method(D_METHOD("get_model_path"), &ExecuTorchMV2Demo::get_model_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path"), "set_model_path", "get_model_path");
}

void ExecuTorchMV2Demo::_ready() {
	if (!model_path_.is_empty()) {
		resource_ = Ref<ExecuTorchResource>(memnew(ExecuTorchResource));
		if (resource_->load_from_file(model_path_) == OK) {
			print_line("Model loaded successfully from: " + model_path_);

			// Run demo inference with dummy input
			PackedFloat32Array input_arr;
			for (int i = 0; i < 1000; ++i) { // Placeholder for image data
				input_arr.push_back(1.0f / (i + 1));
			}

			Array input_data;
			input_data.push_back(input_arr);

			Array output = resource_->forward_array(input_data);
			if (!output.is_empty()) {
				PackedFloat32Array out_arr = output[0];
				if (out_arr.size() > 0) {
					print_line("Inference result: " + rtos(out_arr[0]));
					print_line("Demo completed successfully");
				} else {
					print_line("No output data");
				}
			} else {
				print_line("Inference failed");
			}
		} else {
			print_error("Failed to load model from: " + model_path_);
		}
	} else {
		print_line("No model path set");
	}
}

void ExecuTorchMV2Demo::set_model_path(const String &path) {
	model_path_ = path;
	if (!model_path_.is_empty()) {
		resource_ = Ref<ExecuTorchResource>(memnew(ExecuTorchResource));
	} else {
		resource_ = nullptr;
	}
}

String ExecuTorchMV2Demo::get_model_path() const {
	return model_path_;
}

ExecuTorchMV2Demo::ExecuTorchMV2Demo() {
}

ExecuTorchMV2Demo::~ExecuTorchMV2Demo() {
}
