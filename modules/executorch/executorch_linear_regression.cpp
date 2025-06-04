/**************************************************************************/
/*  executorch_linear_regression.cpp                                      */
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

#include "executorch_linear_regression.h"
#include "core/object/class_db.h"
#include "core/os/time.h"

ExecuTorchLinearRegression::ExecuTorchLinearRegression() :
		slope(2.0),
		intercept(3.0),
		total_inferences_count(0),
		last_inference_time_ms(0.0) {
	_initialize_mcp_tools();
}

ExecuTorchLinearRegression::~ExecuTorchLinearRegression() {
}

void ExecuTorchLinearRegression::_bind_methods() {
	// Linear regression parameters
	ClassDB::bind_method(D_METHOD("set_slope", "slope"), &ExecuTorchLinearRegression::set_slope);
	ClassDB::bind_method(D_METHOD("get_slope"), &ExecuTorchLinearRegression::get_slope);
	ClassDB::bind_method(D_METHOD("set_intercept", "intercept"), &ExecuTorchLinearRegression::set_intercept);
	ClassDB::bind_method(D_METHOD("get_intercept"), &ExecuTorchLinearRegression::get_intercept);

	// Inference methods
	ClassDB::bind_method(D_METHOD("run_inference", "inputs"), &ExecuTorchLinearRegression::run_inference);

	// MCP tools
	ClassDB::bind_method(D_METHOD("list_mcp_tools"), &ExecuTorchLinearRegression::list_mcp_tools);
	ClassDB::bind_method(D_METHOD("get_model_info"), &ExecuTorchLinearRegression::get_model_info);
	ClassDB::bind_method(D_METHOD("health_check"), &ExecuTorchLinearRegression::health_check);
	ClassDB::bind_method(D_METHOD("call_mcp_tool", "tool_name", "arguments"), &ExecuTorchLinearRegression::call_mcp_tool);

	// Performance monitoring
	ClassDB::bind_method(D_METHOD("reset_performance_stats"), &ExecuTorchLinearRegression::reset_performance_stats);
	ClassDB::bind_method(D_METHOD("get_total_inferences"), &ExecuTorchLinearRegression::get_total_inferences);
	ClassDB::bind_method(D_METHOD("get_last_inference_time"), &ExecuTorchLinearRegression::get_last_inference_time);

	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "slope"), "set_slope", "get_slope");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "intercept"), "set_intercept", "get_intercept");

	// Signals
	ADD_SIGNAL(MethodInfo("inference_completed", PropertyInfo(Variant::DICTIONARY, "result")));
}

void ExecuTorchLinearRegression::set_slope(double p_slope) {
	slope = p_slope;
}

double ExecuTorchLinearRegression::get_slope() const {
	return slope;
}

void ExecuTorchLinearRegression::set_intercept(double p_intercept) {
	intercept = p_intercept;
}

double ExecuTorchLinearRegression::get_intercept() const {
	return intercept;
}

Dictionary ExecuTorchLinearRegression::run_inference(const Dictionary &inputs) {
	uint64_t start_time = Time::get_singleton()->get_ticks_usec();

	Dictionary result;

	// Check if input_0 exists
	if (!inputs.has("input_0")) {
		print_error("Missing input_0 in inference inputs");
		return result;
	}

	Variant input_var = inputs["input_0"];
	
	// Handle different input formats
	double input_value = 0.0;
	if (input_var.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
		PackedFloat32Array input_array = input_var;
		if (input_array.size() > 0) {
			input_value = input_array[0];
		}
	} else if (input_var.get_type() == Variant::ARRAY) {
		Array input_array = input_var;
		if (input_array.size() > 0) {
			input_value = input_array[0];
		}
	} else if (input_var.get_type() == Variant::FLOAT) {
		input_value = input_var;
	}

	// Run linear regression: y = slope * x + intercept
	double output_value = slope * input_value + intercept;

	// Create output array
	PackedFloat32Array output_array;
	output_array.push_back(output_value);
	result["output_0"] = output_array;

	uint64_t end_time = Time::get_singleton()->get_ticks_usec();
	double inference_time = (end_time - start_time) / 1000.0; // Convert to milliseconds

	_update_performance_stats(inference_time);

	print_line("Linear regression: f(" + rtos(input_value) + ") = " + rtos(slope) + " * " + rtos(input_value) + " + " + rtos(intercept) + " = " + rtos(output_value));

	emit_signal("inference_completed", result);
	return result;
}

PackedFloat32Array ExecuTorchLinearRegression::predict(const PackedFloat32Array &input) {
	if (input.is_empty()) {
		return PackedFloat32Array();
	}

	Dictionary inputs;
	inputs["input_0"] = input;
	
	Dictionary result = run_inference(inputs);
	
	if (result.has("output_0")) {
		return result["output_0"];
	}
	
	return PackedFloat32Array();
}

Array ExecuTorchLinearRegression::list_mcp_tools() const {
	Array tools;
	Array keys = mcp_tools.keys();
	for (int i = 0; i < keys.size(); i++) {
		tools.push_back(keys[i]);
	}
	return tools;
}

Dictionary ExecuTorchLinearRegression::get_model_info() const {
	Dictionary info;
	info["model_type"] = "linear_regression";
	info["slope"] = slope;
	info["intercept"] = intercept;
	info["equation"] = "y = " + rtos(slope) + " * x + " + rtos(intercept);
	info["input_shape"] = Array();
	info["output_shape"] = Array();
	info["total_inferences"] = total_inferences_count;
	info["last_inference_time_ms"] = last_inference_time_ms;
	return info;
}

Dictionary ExecuTorchLinearRegression::health_check() const {
	Dictionary health;
	health["status"] = "healthy";
	health["model_loaded"] = true;
	health["can_run_inference"] = true;
	health["total_inferences"] = total_inferences_count;
	health["memory_usage"] = "N/A (analytical model)";
	return health;
}

Dictionary ExecuTorchLinearRegression::call_mcp_tool(const String &tool_name, const Dictionary &arguments) {
	Dictionary result;

	if (tool_name == "run_inference") {
		result = run_inference(arguments);
	} else if (tool_name == "get_model_info") {
		result = get_model_info();
	} else if (tool_name == "health_check") {
		result = health_check();
	} else if (tool_name == "reset_stats") {
		reset_performance_stats();
		result["success"] = true;
		result["message"] = "Performance stats reset";
	} else {
		result["error"] = "Unknown tool: " + tool_name;
	}

	return result;
}

void ExecuTorchLinearRegression::reset_performance_stats() {
	total_inferences_count = 0;
	last_inference_time_ms = 0.0;
	print_line("Performance stats reset");
}

int64_t ExecuTorchLinearRegression::get_total_inferences() const {
	return total_inferences_count;
}

double ExecuTorchLinearRegression::get_last_inference_time() const {
	return last_inference_time_ms;
}

void ExecuTorchLinearRegression::_initialize_mcp_tools() {
	mcp_tools.clear();

	Dictionary run_inference_tool;
	run_inference_tool["name"] = "run_inference";
	run_inference_tool["description"] = "Run linear regression inference on input data";
	mcp_tools["run_inference"] = run_inference_tool;

	Dictionary model_info_tool;
	model_info_tool["name"] = "get_model_info";
	model_info_tool["description"] = "Get information about the linear regression model";
	mcp_tools["get_model_info"] = model_info_tool;

	Dictionary health_tool;
	health_tool["name"] = "health_check";
	health_tool["description"] = "Check the health status of the model";
	mcp_tools["health_check"] = health_tool;

	Dictionary reset_tool;
	reset_tool["name"] = "reset_stats";
	reset_tool["description"] = "Reset performance statistics";
	mcp_tools["reset_stats"] = reset_tool;

	print_line("MCP tools initialized for linear regression");
}

Dictionary ExecuTorchLinearRegression::_run_linear_regression(double input_value) const {
	Dictionary result;
	double output_value = slope * input_value + intercept;
	
	PackedFloat32Array output_array;
	output_array.push_back(output_value);
	result["output_0"] = output_array;
	
	return result;
}

void ExecuTorchLinearRegression::_update_performance_stats(double inference_time) const {
	last_inference_time_ms = inference_time;
	total_inferences_count++;
}