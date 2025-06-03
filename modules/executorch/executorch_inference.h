/**************************************************************************/
/*  executorch_inference.h                                                */
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

#include "executorch_model.h"
#include "executorch_runtime.h"
#include <memory>

// Convenience wrapper for simple inference use cases
class ExecuTorchInference {
private:
	std::unique_ptr<ExecuTorchRuntime> runtime_;
	std::unique_ptr<ExecuTorchModel> model_;
	bool auto_manage_runtime_;

public:
	ExecuTorchInference(bool auto_manage = true);
	~ExecuTorchInference();

	// Simple API for quick setup
	bool load_model(const std::string &file_path);
	std::vector<float> predict(const std::vector<float> &input);

	// Advanced API for when you need more control
	ExecuTorchRuntime *get_runtime() { return runtime_.get(); }
	ExecuTorchModel *get_model() { return model_.get(); }

	// Allow using external runtime (for sharing between models)
	void set_runtime(ExecuTorchRuntime *external_runtime);
};
