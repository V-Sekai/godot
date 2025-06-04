/**************************************************************************/
/*  executorch_linear_regression.h                                        */
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

#include "executorch_node.h"

class ExecuTorchLinearRegression : public ExecuTorchNode {
	GDCLASS(ExecuTorchLinearRegression, ExecuTorchNode);

private:
	// Linear regression parameters: y = slope * x + intercept
	double slope;
	double intercept;
	
	// Performance tracking
	mutable int64_t total_inferences_count;
	mutable double last_inference_time_ms;
	
	// MCP integration
	Dictionary mcp_tools;
	Dictionary model_info_cache;

protected:
	static void _bind_methods();

public:
	ExecuTorchLinearRegression();
	~ExecuTorchLinearRegression();

	// Linear regression specific methods
	void set_slope(double p_slope);
	double get_slope() const;
	void set_intercept(double p_intercept);
	double get_intercept() const;
	
	// Override inference methods
	Dictionary run_inference(const Dictionary &inputs);
	PackedFloat32Array predict(const PackedFloat32Array &input) override;
	
	// MCP tools interface
	Array list_mcp_tools() const;
	Dictionary get_model_info() const;
	Dictionary health_check() const;
	Dictionary call_mcp_tool(const String &tool_name, const Dictionary &arguments);
	
	// Performance monitoring
	void reset_performance_stats();
	int64_t get_total_inferences() const;
	double get_last_inference_time() const;

private:
	void _initialize_mcp_tools();
	Dictionary _run_linear_regression(double input_value) const;
	void _update_performance_stats(double inference_time) const;
};