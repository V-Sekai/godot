/**************************************************************************/
/*  executorch_inference.cpp                                             */
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
#include <iostream>

ExecuTorchInference::ExecuTorchInference(bool auto_manage) : auto_manage_runtime_(auto_manage) {
    if (auto_manage_runtime_) {
        runtime_ = std::make_unique<ExecuTorchRuntime>();
    }
    model_ = std::make_unique<ExecuTorchModel>();
}

ExecuTorchInference::~ExecuTorchInference() {
    // Unique pointers will automatically clean up
}

bool ExecuTorchInference::load_model(const std::string &file_path) {
    if (!model_) {
        std::cerr << "Model not initialized" << std::endl;
        return false;
    }
    
    // Initialize runtime if we're managing it
    if (auto_manage_runtime_ && runtime_) {
        if (!runtime_->initialize()) {
            std::cerr << "Failed to initialize ExecuTorch runtime" << std::endl;
            return false;
        }
    }
    
    // Load the model
    bool success = model_->load_from_file(file_path);
    if (!success) {
        std::cerr << "Failed to load model from: " << file_path << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded model: " << file_path << std::endl;
    return true;
}

std::vector<float> ExecuTorchInference::predict(const std::vector<float> &input) {
    if (!model_ || !model_->is_loaded()) {
        std::cerr << "Model not loaded" << std::endl;
        return {};
    }
    
    // Use the single input forward method for simplicity
    return model_->forward_single(input);
}

void ExecuTorchInference::set_runtime(ExecuTorchRuntime* external_runtime) {
    if (auto_manage_runtime_) {
        // Release our managed runtime
        runtime_.reset();
        auto_manage_runtime_ = false;
    }
    
    // Note: We're not storing the external runtime pointer here
    // In a real implementation, you'd need to modify ExecuTorchModel 
    // to accept and use the external runtime
    std::cout << "External runtime set (implementation pending)" << std::endl;
}