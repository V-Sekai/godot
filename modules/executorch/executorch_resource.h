/**************************************************************************/
/*  executorch_resource.h                                                 */
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
#include <cstdint>
#include <memory>
#include <vector>

class ExecuTorchModule;
class ExecuTorchMemoryManager;

/**
 * ExecuTorchResource - A Godot Resource for .pte (PyTorch ExecuTorch) files
 *
 * This resource class provides both high-level and low-level APIs for ExecuTorch models:
 * - High-level: Simple forward() method using ExecuTorch Module class
 * - Low-level: Direct memory management and placement control
 */
class ExecuTorchResource : public RefCounted {
public:
	enum MemoryPolicy {
		MEMORY_POLICY_AUTO, // Automatic memory management
		MEMORY_POLICY_STATIC, // Static memory allocation
		MEMORY_POLICY_CUSTOM // Custom memory allocator
	};

	enum OptimizationLevel {
		OPTIMIZATION_NONE = 0,
		OPTIMIZATION_BASIC = 1,
		OPTIMIZATION_AGGRESSIVE = 2
	};

private:
	// Core model data
	PackedByteArray model_data_;
	String source_file_path_;
	bool is_loaded_;

	// ExecuTorch components
	std::unique_ptr<ExecuTorchModule> module_;
	std::unique_ptr<ExecuTorchMemoryManager> memory_manager_;

	// Configuration
	MemoryPolicy memory_policy_;
	OptimizationLevel optimization_level_;
	int64_t memory_limit_bytes_;
	bool enable_profiling_;

	// Model metadata
	Array input_names_;
	Array output_names_;
	Dictionary input_shapes_;
	Dictionary output_shapes_;
	String model_name_;
	String model_version_;

	// Performance tracking
	mutable double last_inference_time_ms_;
	mutable int total_inferences_;

public:
	ExecuTorchResource();
	virtual ~ExecuTorchResource();

	// Resource interface
	virtual Error load_from_file(const String &path);
	virtual Error save_to_file(const String &path);
	virtual void clear();

	// High-level API (using ExecuTorch Module class)
	Dictionary forward(const Dictionary &inputs);
	Array forward_array(const Array &input_data);

	// Low-level API (direct ExecuTorch control)
	Error configure_memory(MemoryPolicy policy, int64_t limit_bytes = 0);
	Error set_optimization_level(OptimizationLevel level);
	Error enable_profiling(bool enable);

	// Model metadata
	Array get_input_names() const { return input_names_; }
	Array get_output_names() const { return output_names_; }
	Dictionary get_input_shapes() const { return input_shapes_; }
	Dictionary get_output_shapes() const { return output_shapes_; }
	String get_model_name() const { return model_name_; }
	String get_model_version() const { return model_version_; }

	// Status and diagnostics
	bool is_loaded() const { return is_loaded_; }
	int64_t get_model_size() const { return model_data_.size(); }
	double get_last_inference_time() const { return last_inference_time_ms_; }
	int get_total_inferences() const { return total_inferences_; }
	Dictionary get_memory_info() const;

	// Data access
	PackedByteArray get_model_data() const { return model_data_; }
	void set_model_data(const PackedByteArray &data);
	String get_source_file_path() const { return source_file_path_; }

private:
	// Internal implementation
	Error _load_with_high_level_api();
	Error _load_with_low_level_api();
	void _extract_metadata();
	void _update_performance_stats(double inference_time) const;
	Dictionary _convert_tensors_to_dictionary(const std::vector<void *> &tensors, const Array &names) const;
	std::vector<void *> _convert_dictionary_to_tensors(const Dictionary &inputs) const;
};

/**
 * ExecuTorch Module wrapper - High-level API
 */
// TODO: Extends Node and takes a PTE resource.
class ExecuTorchModule {
private:
	bool is_loaded_;
	String file_path_;
	PackedByteArray buffer_data_;
	void *native_module_; // Actual ExecuTorch Module pointer

public:
	ExecuTorchModule();
	~ExecuTorchModule();

	// High-level interface matching ExecuTorch C++ Module class
	Error load(const String &file_path);
	Error load_from_buffer(const PackedByteArray &buffer);
	Dictionary forward(const Dictionary &inputs);
	void unload();
	bool is_loaded() const { return is_loaded_; }

	// Metadata access
	Array get_method_names() const;
	Dictionary get_method_meta(const String &method_name = "forward") const;
};

/**
 * ExecuTorch Memory Manager - Low-level memory control
 */
class ExecuTorchMemoryManager {
private:
	void *memory_allocator_;
	void *memory_pool_;
	size_t pool_size_;
	bool is_static_allocation_;

public:
	ExecuTorchMemoryManager();
	~ExecuTorchMemoryManager();

	Error configure_static_memory(size_t pool_size);
	Error configure_dynamic_memory();
	Error configure_custom_allocator(void *allocator);

	Dictionary get_memory_stats() const;
	size_t get_allocated_bytes() const;
	size_t get_available_bytes() const;

	void *allocate(size_t size, size_t alignment = 16);
	void deallocate(void *ptr);
	void reset();
};
