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

#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward declarations
class ExecuTorchModel;
class MCPServerInternal;

// Simplified types for standalone compilation
using String = std::string;
using Dictionary = std::map<std::string, std::vector<float>>;
using Array = std::vector<float>;
using PackedByteArray = std::vector<uint8_t>;

class ExecuTorchRuntime {
private:
	std::shared_ptr<ExecuTorchModel> model_;
	std::unique_ptr<MCPServerInternal> mcp_server_;
	bool is_initialized_;
	String model_path_;

	// Performance metrics
	double last_inference_time_ms_;
	int total_inferences_;

public:
	ExecuTorchRuntime();
	virtual ~ExecuTorchRuntime();

	// Core model management
	bool load_model_from_pck(const String &pck_path);
	bool load_model_from_file(const String &file_path);
	void unload_model();
	bool is_model_loaded() const;

	// Inference methods
	Dictionary run_inference(const Dictionary &inputs);
	Array run_inference_array(const Array &input_data);

	// Performance and diagnostics
	double get_last_inference_time() const;
	int get_total_inferences() const;
	void reset_performance_stats();

	// Configuration
	void set_optimization_level(int level);
	void enable_profiling(bool enable);
	void set_memory_limit(int64_t bytes);

	// MCP tools interface
	std::vector<String> list_mcp_tools() const;
	Dictionary call_mcp_tool(const String &tool_name, const Dictionary &params);
	Dictionary get_model_info() const;
	Dictionary health_check() const;

private:
	// Internal helpers
	bool _load_model_from_buffer(const PackedByteArray &model_data);
	void _update_performance_stats(double inference_time);
	Dictionary _convert_cpp_result(const std::map<std::string, std::vector<float>> &cpp_result);
	std::map<std::string, std::vector<float>> _convert_godot_inputs(const Dictionary &godot_inputs);
};
