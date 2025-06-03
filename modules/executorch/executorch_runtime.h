/**************************************************************************/
/*  executorch_runtime.h                                                  */
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

#include <cstddef> // for size_t

enum class ExecuTorchDevice {
	CPU,
	CUDA,
	METAL,
	VULKAN
};

class ExecuTorchRuntime {
private:
	bool is_initialized_;
	ExecuTorchDevice device_;
	size_t memory_pool_size_;
	int num_threads_;

public:
	ExecuTorchRuntime();
	~ExecuTorchRuntime();

	// Runtime management
	bool initialize();
	void shutdown();
	bool is_initialized() const { return is_initialized_; }

	// Configuration
	void set_device(ExecuTorchDevice device) { device_ = device; }
	ExecuTorchDevice get_device() const { return device_; }
	void set_memory_pool_size(size_t size) { memory_pool_size_ = size; }
	size_t get_memory_pool_size() const { return memory_pool_size_; }
	void set_num_threads(int threads) { num_threads_ = threads; }
	int get_num_threads() const { return num_threads_; }

	// Memory management
	void *allocate_memory(size_t size);
	void deallocate_memory(void *ptr);
	void clear_memory_pool();

	// Performance monitoring
	double get_last_inference_time() const;
	size_t get_memory_usage() const;

private:
	bool _initialize_device();
	bool _setup_memory_pool();
	bool _configure_threading();
};
