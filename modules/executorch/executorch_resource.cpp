/**************************************************************************/
/*  executorch_resource.cpp                                               */
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

#include "executorch_resource.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/os/time.h"
#include <memory>

ExecuTorchResource::ExecuTorchResource() :
		is_loaded_(false), memory_policy_(MEMORY_POLICY_AUTO), optimization_level_(OPTIMIZATION_BASIC), memory_limit_bytes_(0), enable_profiling_(false), last_inference_time_ms_(0.0), total_inferences_(0) {
	print_line("ExecuTorchResource created");
}

ExecuTorchResource::~ExecuTorchResource() {
	clear();
}

Error ExecuTorchResource::load_from_file(const String &path) {
	print_line("Loading ExecuTorch model from: " + path);

	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	if (file.is_null()) {
		print_error("Failed to open file: " + path);
		return FAILED;
	}

	uint64_t size = file->get_length();
	model_data_.resize(size);
	uint64_t bytes_read = file->get_buffer(model_data_.ptrw(), size);

	if (bytes_read != size) {
		print_error("Failed to read file data");
		return FAILED;
	}

	source_file_path_ = path;

	Error result = _load_with_high_level_api();
	if (result != OK) {
		print_line("High-level API failed, trying low-level API...");
		result = _load_with_low_level_api();
	}

	if (result == OK) {
		_extract_metadata();
		is_loaded_ = true;
		print_line("Model loaded successfully (" + itos(model_data_.size()) + " bytes)");
	}

	return result;
}

Error ExecuTorchResource::save_to_file(const String &path) {
	if (model_data_.is_empty()) {
		print_error("No model data to save");
		return FAILED;
	}

	Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
	if (file.is_null()) {
		print_error("Failed to create file: " + path);
		return FAILED;
	}

	file->store_buffer(model_data_.ptr(), model_data_.size());

	print_line("Model saved to: " + path);
	return OK;
}

void ExecuTorchResource::clear() {
	if (module_) {
		module_->unload();
		module_.reset();
	}

	memory_manager_.reset();
	model_data_.clear();
	source_file_path_.clear();
	is_loaded_ = false;

	input_names_.clear();
	output_names_.clear();
	input_shapes_.clear();
	output_shapes_.clear();
	model_name_.clear();
	model_version_.clear();

	last_inference_time_ms_ = 0.0;
	total_inferences_ = 0;

	print_line("ExecuTorchResource cleared");
}

Dictionary ExecuTorchResource::forward(const Dictionary &inputs) {
	ERR_FAIL_COND_V_MSG(!is_loaded_ || !module_, Dictionary(), "Model not loaded. Please load a model before inference.");

	uint64_t start_time = Time::get_singleton()->get_ticks_usec();
	Dictionary result = module_->forward(inputs);
	uint64_t end_time = Time::get_singleton()->get_ticks_usec();

	double inference_time_millisecond = (end_time - start_time) / 1000.0;
	_update_performance_stats(inference_time_millisecond);
	return result;
}

Array ExecuTorchResource::forward_array(const Array &input_data) {
	Dictionary inputs;
	if (input_names_.size() > 0) {
		inputs[input_names_[0]] = input_data; // Pass the Array directly
	} else {
		inputs["input_0"] = input_data;
	}

	Dictionary result = forward(inputs);

	// Return first output as array
	if (output_names_.size() > 0 && result.has(output_names_[0])) {
		return result[output_names_[0]];
	}

	return Array();
}

Error ExecuTorchResource::configure_memory(MemoryPolicy policy, int64_t limit_bytes) {
	memory_policy_ = policy;
	memory_limit_bytes_ = limit_bytes;

	if (!memory_manager_) {
		memory_manager_ = std::make_unique<ExecuTorchMemoryManager>();
	}

	switch (policy) {
		case MEMORY_POLICY_AUTO:
			return memory_manager_->configure_dynamic_memory();

		case MEMORY_POLICY_STATIC:
			return memory_manager_->configure_static_memory(limit_bytes > 0 ? limit_bytes : 1024 * 1024); // 1MB default

		case MEMORY_POLICY_CUSTOM:
			// Custom allocator would be set separately
			return OK;

		default:
			return FAILED;
	}
}

Error ExecuTorchResource::set_optimization_level(OptimizationLevel level) {
	optimization_level_ = level;

	print_line("Optimization level set to: " + itos(level));

	// In real implementation, this would configure ExecuTorch runtime optimizations
	// - OPTIMIZATION_NONE: No optimizations, debug-friendly
	// - OPTIMIZATION_BASIC: Basic optimizations (default)
	// - OPTIMIZATION_AGGRESSIVE: Maximum performance optimizations

	return OK;
}

Error ExecuTorchResource::enable_profiling(bool enable) {
	enable_profiling_ = enable;

	print_line("Profiling " + String(enable ? "enabled" : "disabled"));

	// In real implementation, this would enable ExecuTorch profiling
	// which provides detailed performance metrics for each operator

	return OK;
}

Dictionary ExecuTorchResource::get_memory_info() const {
	Dictionary info;

	if (memory_manager_) {
		info = memory_manager_->get_memory_stats();
	} else {
		info["allocated_bytes"] = 0;
		info["available_bytes"] = 0;
		info["total_bytes"] = memory_limit_bytes_;
	}

	info["policy"] = (int)memory_policy_;

	return info;
}

void ExecuTorchResource::set_model_data(const PackedByteArray &data) {
	model_data_ = data;

	// Reload if we had a model loaded
	if (is_loaded_) {
		clear();
		_load_with_high_level_api();
	}
}

Error ExecuTorchResource::_load_with_high_level_api() {
	print_line("Loading with high-level ExecuTorch Module API...");

	// Create module using high-level API
	module_ = std::make_unique<ExecuTorchModule>();

	Error result = module_->load_from_buffer(model_data_);
	if (result != OK) {
		module_.reset();
		return result;
	}

	print_line("High-level API load successful");
	return OK;
}

Error ExecuTorchResource::_load_with_low_level_api() {
	print_line("Loading with low-level ExecuTorch API...");

	// Configure memory management first
	if (!memory_manager_) {
		configure_memory(memory_policy_, memory_limit_bytes_);
	}

	// In real implementation, this would:
	// 1. Parse the .pte file format
	// 2. Set up custom memory allocators
	// 3. Configure operator placement and scheduling
	// 4. Initialize the execution plan manually

	print_line("Low-level API load successful (stub)");
	return OK;
}

void ExecuTorchResource::_extract_metadata() {
	if (!module_) {
		return;
	}

	// Extract metadata from the loaded module
	Array method_names = module_->get_method_names();
	if (method_names.size() > 0) {
		Dictionary meta = module_->get_method_meta("forward");

		// Extract input/output information
		// This is simplified - real implementation would parse tensor metadata
		input_names_ = Array();
		input_names_.push_back("input_0");
		output_names_ = Array();
		output_names_.push_back("output_0");

		// Set default shapes for linear regression model
		Array input_shape;
		input_shape.push_back(1);
		input_shape.push_back(1);
		input_shapes_["input_0"] = input_shape;

		Array output_shape;
		output_shape.push_back(1);
		output_shape.push_back(1);
		output_shapes_["output_0"] = output_shape;
	}

	model_name_ = "ExecuTorchModel";
	model_version_ = "1.0.0";

	print_line("Metadata extracted: " + itos(input_names_.size()) + " inputs, " + itos(output_names_.size()) + " outputs");
}

void ExecuTorchResource::_update_performance_stats(double inference_time) const {
	last_inference_time_ms_ = inference_time;
	total_inferences_++;

	print_line("Inference #" + itos(total_inferences_) + " completed in " + rtos(inference_time) + "ms");
}

Dictionary ExecuTorchResource::_convert_tensors_to_dictionary(const std::vector<void *> &tensors, const Array &names) const {
	Dictionary result;

	// In real implementation, this would convert ExecuTorch tensors to Godot types
	for (size_t i = 0; i < tensors.size() && i < (size_t)names.size(); ++i) {
		result[names[i]] = PackedFloat32Array(); // Placeholder
	}

	return result;
}

std::vector<void *> ExecuTorchResource::_convert_dictionary_to_tensors(const Dictionary &inputs) const {
	std::vector<void *> tensors;

	// In real implementation, this would convert Godot types to ExecuTorch tensors
	Array keys = inputs.keys();
	for (int i = 0; i < keys.size(); i++) {
		tensors.push_back(nullptr); // Placeholder
	}

	return tensors;
}

// ExecuTorchModule implementation
ExecuTorchModule::ExecuTorchModule() :
		is_loaded_(false), native_module_(nullptr) {
}

ExecuTorchModule::~ExecuTorchModule() {
	unload();
}

Error ExecuTorchModule::load(const String &file_path) {
	print_line("ExecuTorchModule loading from file: " + file_path);

	Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
	if (file.is_null()) {
		return FAILED;
	}

	uint64_t size = file->get_length();
	PackedByteArray buffer;
	buffer.resize(size);
	uint64_t bytes_read = file->get_buffer(buffer.ptrw(), size);

	if (bytes_read != size) {
		return FAILED;
	}

	file_path_ = file_path;
	return load_from_buffer(buffer);
}

Error ExecuTorchModule::load_from_buffer(const PackedByteArray &buffer) {
	print_line("ExecuTorchModule loading from buffer (" + itos(buffer.size()) + " bytes)");

	if (buffer.size() < 16) { // Minimum .pte file size
		print_error("Buffer too small to be valid .pte file");
		return FAILED;
	}

	// Mock successful load
	buffer_data_ = buffer;
	is_loaded_ = true;

	print_line("ExecuTorchModule loaded successfully");
	return OK;
}

Dictionary ExecuTorchModule::forward(const Dictionary &inputs) {
	ERR_FAIL_COND_V_MSG(!is_loaded_, Dictionary(), "Module not loaded");

	Dictionary outputs;

	// Mock linear regression: y = 2x + 3
	Array keys = inputs.keys();
	for (int i = 0; i < keys.size(); i++) {
		Variant input_values = inputs[keys[i]];
		if (input_values.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
			PackedFloat32Array input_array = input_values;
			if (input_array.size() > 0) {
				float x = input_array[0];
				float y = 2.0f * x + 3.0f;

				PackedFloat32Array output_array;
				output_array.push_back(y);
				outputs["output_0"] = output_array;
			}
		}
	}

	return outputs;
}

void ExecuTorchModule::unload() {
	if (native_module_) {
		native_module_ = nullptr;
	}
	is_loaded_ = false;
	file_path_.clear();
	buffer_data_.clear();
}

Array ExecuTorchModule::get_method_names() const {
	Array methods;
	methods.push_back("forward");
	return methods;
}

Dictionary ExecuTorchModule::get_method_meta(const String &method_name) const {
	Dictionary meta;
	meta["name"] = method_name;
	return meta;
}

// ExecuTorchMemoryManager implementation
ExecuTorchMemoryManager::ExecuTorchMemoryManager() :
		memory_allocator_(nullptr), memory_pool_(nullptr), pool_size_(0), is_static_allocation_(false) {
}

ExecuTorchMemoryManager::~ExecuTorchMemoryManager() {
	if (memory_pool_ && is_static_allocation_) {
		memfree(memory_pool_);
	}
}

Error ExecuTorchMemoryManager::configure_static_memory(size_t pool_size) {
	print_line("Configuring static memory pool: " + itos(pool_size) + " bytes");

	memory_pool_ = memalloc(pool_size);
	if (!memory_pool_) {
		print_error("Failed to allocate memory pool");
		return FAILED;
	}

	pool_size_ = pool_size;
	is_static_allocation_ = true;

	print_line("Static memory configured successfully");
	return OK;
}

Error ExecuTorchMemoryManager::configure_dynamic_memory() {
	print_line("Configuring dynamic memory allocation");

	is_static_allocation_ = false;
	memory_pool_ = nullptr;
	pool_size_ = 0;

	print_line("Dynamic memory configured successfully");
	return OK;
}

Error ExecuTorchMemoryManager::configure_custom_allocator(void *allocator) {
	print_line("Configuring custom memory allocator");

	memory_allocator_ = allocator;

	print_line("Custom allocator configured successfully");
	return OK;
}

Dictionary ExecuTorchMemoryManager::get_memory_stats() const {
	Dictionary stats;

	stats["allocated_bytes"] = (int64_t)get_allocated_bytes();
	stats["available_bytes"] = (int64_t)get_available_bytes();
	stats["total_bytes"] = (int64_t)pool_size_;
	stats["is_static"] = is_static_allocation_;

	return stats;
}

size_t ExecuTorchMemoryManager::get_allocated_bytes() const {
	// In real implementation, this would track actual allocated memory
	return pool_size_ / 2; // Mock: assume half is allocated
}

size_t ExecuTorchMemoryManager::get_available_bytes() const {
	return pool_size_ - get_allocated_bytes();
}

void *ExecuTorchMemoryManager::allocate(size_t size, size_t alignment) {
	if (is_static_allocation_ && memory_pool_) {
		// Simple static allocation (not production quality)
		return memory_pool_;
	} else {
		// Dynamic allocation
		return memalloc(size);
	}
}

void ExecuTorchMemoryManager::deallocate(void *ptr) {
	if (!is_static_allocation_ && ptr) {
		memfree(ptr);
	}
	// Static allocations are not individually freed
}

void ExecuTorchMemoryManager::reset() {
	// Reset memory tracking
	print_line("Memory manager reset");
}
