/**************************************************************************/
/*  executorch_model.h                                                    */
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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ExecuTorchModel {
private:
	bool is_loaded_;
	std::map<std::string, std::vector<int64_t>> input_shapes_;
	std::map<std::string, std::vector<int64_t>> output_shapes_;
	std::vector<std::string> input_names_;
	std::vector<std::string> output_names_;

public:
	ExecuTorchModel();
	~ExecuTorchModel();

	// Core loading functionality
	bool load_from_file(const std::string &file_path);
	bool load_from_buffer(const std::vector<uint8_t> &model_data);
	void unload();
	bool is_loaded() const { return is_loaded_; }

	// Model metadata
	std::vector<std::string> get_input_names() const { return input_names_; }
	std::vector<std::string> get_output_names() const { return output_names_; }
	std::vector<int64_t> get_input_shape(const std::string &name) const;
	std::vector<int64_t> get_output_shape(const std::string &name) const;

	// Inference
	std::map<std::string, std::vector<float>> forward(const std::map<std::string, std::vector<float>> &inputs);
	std::vector<float> forward_single(const std::vector<float> &input);
	std::map<std::string, std::vector<float>> forward_named(const std::map<std::string, std::vector<float>> &inputs);

private:
	bool _initialize_metadata();
};
