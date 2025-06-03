/**************************************************************************/
/*  mcp_server_internal.cpp                                               */
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

#include "mcp_server_internal.h"
#include "executorch_model.h"
#include <iostream>

MCPServerInternal::MCPServerInternal() :
		is_initialized_(false), server_name_("ExecuTorchMCPServer"), server_version_("1.0.0") {
}

MCPServerInternal::~MCPServerInternal() {
	shutdown();
}

bool MCPServerInternal::initialize(const std::string &name, const std::string &version) {
	if (is_initialized_) {
		std::cerr << "MCP Server already initialized" << std::endl;
		return false;
	}

	server_name_ = name;
	server_version_ = version;

	// Register built-in tools
	_register_builtin_tools();

	is_initialized_ = true;
	std::cout << "MCP Server '" << server_name_ << "' v" << server_version_ << " initialized" << std::endl;

	return true;
}

void MCPServerInternal::shutdown() {
	if (!is_initialized_) {
		return;
	}

	tools_.clear();
	handlers_.clear();
	model_.reset();
	is_initialized_ = false;

	std::cout << "MCP Server shutdown" << std::endl;
}

bool MCPServerInternal::set_model(std::shared_ptr<ExecuTorchModel> model) {
	if (!model) {
		std::cerr << "Cannot set null model" << std::endl;
		return false;
	}

	if (!model->is_loaded()) {
		std::cerr << "Cannot set unloaded model" << std::endl;
		return false;
	}

	model_ = model;
	std::cout << "Model set in MCP Server" << std::endl;
	return true;
}

bool MCPServerInternal::register_tool(const MCPToolDefinition &tool_def,
		std::function<MCPResponse(const MCPRequest &)> handler) {
	if (tools_.find(tool_def.name) != tools_.end()) {
		std::cerr << "Tool '" << tool_def.name << "' already registered" << std::endl;
		return false;
	}

	tools_[tool_def.name] = tool_def;
	handlers_[tool_def.name] = handler;

	std::cout << "Registered MCP tool: " << tool_def.name << std::endl;
	return true;
}

bool MCPServerInternal::unregister_tool(const std::string &tool_name) {
	auto tool_it = tools_.find(tool_name);
	auto handler_it = handlers_.find(tool_name);

	if (tool_it == tools_.end() || handler_it == handlers_.end()) {
		std::cerr << "Tool '" << tool_name << "' not found" << std::endl;
		return false;
	}

	tools_.erase(tool_it);
	handlers_.erase(handler_it);

	std::cout << "Unregistered MCP tool: " << tool_name << std::endl;
	return true;
}

std::vector<std::string> MCPServerInternal::list_tools() const {
	std::vector<std::string> tool_names;
	for (const auto &[name, _] : tools_) {
		tool_names.push_back(name);
	}
	return tool_names;
}

MCPToolDefinition MCPServerInternal::get_tool_definition(const std::string &tool_name) const {
	auto it = tools_.find(tool_name);
	if (it != tools_.end()) {
		return it->second;
	}
	return MCPToolDefinition{}; // Return empty definition if not found
}

MCPResponse MCPServerInternal::handle_request(const MCPRequest &request) {
	if (!is_initialized_) {
		return _create_error_response("Server not initialized", -32002, request.id);
	}

	if (!_validate_request(request)) {
		return _create_error_response("Invalid request", -32600, request.id);
	}

	auto handler_it = handlers_.find(request.tool_name);
	if (handler_it == handlers_.end()) {
		return _create_error_response("Tool not found: " + request.tool_name, -32601, request.id);
	}

	try {
		return handler_it->second(request);
	} catch (const std::exception &e) {
		return _create_error_response("Internal error: " + std::string(e.what()), -32603, request.id);
	}
}

MCPResponse MCPServerInternal::call_tool(const std::string &tool_name,
		const std::map<std::string, std::vector<float>> &params) {
	MCPRequest request;
	request.method = "tools/call";
	request.tool_name = tool_name;
	request.params = params;
	request.id = 1; // Simple ID for direct calls

	return handle_request(request);
}

MCPResponse MCPServerInternal::handle_inference_tool(const MCPRequest &request) {
	if (!model_) {
		return _create_error_response("No model loaded", -32000, request.id);
	}

	try {
		auto outputs = model_->forward(request.params);
		return _create_success_response(outputs, request.id);
	} catch (const std::exception &e) {
		return _create_error_response("Inference failed: " + std::string(e.what()), -32000, request.id);
	}
}

MCPResponse MCPServerInternal::handle_model_info_tool(const MCPRequest &request) {
	if (!model_) {
		return _create_error_response("No model loaded", -32000, request.id);
	}

	try {
		std::map<std::string, std::vector<float>> info;

		// Add model metadata as float vectors (simplified for demo)
		// In real implementation, this would return proper JSON-structured data
		auto input_names = model_->get_input_names();
		auto output_names = model_->get_output_names();

		// Example: return number of inputs/outputs as floats
		info["num_inputs"] = { static_cast<float>(input_names.size()) };
		info["num_outputs"] = { static_cast<float>(output_names.size()) };

		return _create_success_response(info, request.id);
	} catch (const std::exception &e) {
		return _create_error_response("Failed to get model info: " + std::string(e.what()), -32000, request.id);
	}
}

MCPResponse MCPServerInternal::handle_health_check_tool(const MCPRequest &request) {
	std::map<std::string, std::vector<float>> health_status;

	// Simple health indicators as floats
	health_status["server_initialized"] = { is_initialized_ ? 1.0f : 0.0f };
	health_status["model_loaded"] = { (model_ && model_->is_loaded()) ? 1.0f : 0.0f };
	health_status["num_tools"] = { static_cast<float>(tools_.size()) };

	return _create_success_response(health_status, request.id);
}

void MCPServerInternal::_register_builtin_tools() {
	// Register inference tool
	MCPToolDefinition inference_tool;
	inference_tool.name = "run_inference";
	inference_tool.description = "Execute model inference with provided inputs";
	inference_tool.input_schema["type"] = "object";
	inference_tool.output_schema["type"] = "object";

	register_tool(inference_tool, [this](const MCPRequest &req) {
		return handle_inference_tool(req);
	});

	// Register model info tool
	MCPToolDefinition info_tool;
	info_tool.name = "get_model_info";
	info_tool.description = "Get information about the loaded model";
	info_tool.input_schema["type"] = "object";
	info_tool.output_schema["type"] = "object";

	register_tool(info_tool, [this](const MCPRequest &req) {
		return handle_model_info_tool(req);
	});

	// Register health check tool
	MCPToolDefinition health_tool;
	health_tool.name = "health_check";
	health_tool.description = "Check server and model health status";
	health_tool.input_schema["type"] = "object";
	health_tool.output_schema["type"] = "object";

	register_tool(health_tool, [this](const MCPRequest &req) {
		return handle_health_check_tool(req);
	});
}

MCPResponse MCPServerInternal::_create_error_response(const std::string &message, int code, int request_id) {
	MCPResponse response;
	response.success = false;
	response.error_message = message;
	response.error_code = code;
	response.id = request_id;
	return response;
}

MCPResponse MCPServerInternal::_create_success_response(const std::map<std::string, std::vector<float>> &result, int request_id) {
	MCPResponse response;
	response.success = true;
	response.result = result;
	response.error_code = 0;
	response.id = request_id;
	return response;
}

bool MCPServerInternal::_validate_request(const MCPRequest &request) {
	// Basic validation
	if (request.method.empty()) {
		return false;
	}

	if (request.method == "tools/call" && request.tool_name.empty()) {
		return false;
	}

	return true;
}
