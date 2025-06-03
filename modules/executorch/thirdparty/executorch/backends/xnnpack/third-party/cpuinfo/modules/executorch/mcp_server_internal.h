/**************************************************************************/
/*  mcp_server_internal.h                                                 */
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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
class ExecuTorchModel;

// MCP Protocol structures
struct MCPToolDefinition {
	std::string name;
	std::string description;
	std::map<std::string, std::string> input_schema;
	std::map<std::string, std::string> output_schema;
};

struct MCPRequest {
	std::string method;
	std::string tool_name;
	std::map<std::string, std::vector<float>> params;
	int id;
};

struct MCPResponse {
	bool success;
	std::map<std::string, std::vector<float>> result;
	std::string error_message;
	int error_code;
	int id;
};

class MCPServerInternal {
private:
	bool is_initialized_;
	std::string server_name_;
	std::string server_version_;

	std::shared_ptr<ExecuTorchModel> model_;
	std::map<std::string, MCPToolDefinition> tools_;
	std::map<std::string, std::function<MCPResponse(const MCPRequest &)>> handlers_;

public:
	MCPServerInternal();
	~MCPServerInternal();

	// Server lifecycle
	bool initialize(const std::string &name, const std::string &version);
	void shutdown();
	bool is_initialized() const { return is_initialized_; }

	// Model management
	bool set_model(std::shared_ptr<ExecuTorchModel> model);
	std::shared_ptr<ExecuTorchModel> get_model() const { return model_; }

	// Tool management
	bool register_tool(const MCPToolDefinition &tool_def,
			std::function<MCPResponse(const MCPRequest &)> handler);
	bool unregister_tool(const std::string &tool_name);
	std::vector<std::string> list_tools() const;
	MCPToolDefinition get_tool_definition(const std::string &tool_name) const;

	// Request handling
	MCPResponse handle_request(const MCPRequest &request);
	MCPResponse call_tool(const std::string &tool_name,
			const std::map<std::string, std::vector<float>> &params);

	// Built-in tool handlers
	MCPResponse handle_inference_tool(const MCPRequest &request);
	MCPResponse handle_model_info_tool(const MCPRequest &request);
	MCPResponse handle_health_check_tool(const MCPRequest &request);

private:
	void _register_builtin_tools();
	MCPResponse _create_error_response(const std::string &message, int code, int request_id);
	MCPResponse _create_success_response(const std::map<std::string, std::vector<float>> &result, int request_id);
	bool _validate_request(const MCPRequest &request);
};
