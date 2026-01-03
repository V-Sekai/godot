/**************************************************************************/
/*  mcp_server.cpp                                                        */
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

#include "mcp_server.h"
#include "script_server.h"

#include "core/io/json.h"
#include "core/variant/variant_utility.h"

MCPServer::MCPServer() {
	transport = Ref<HTTPStreamTransport>(memnew(HTTPStreamTransport));
	jsonrpc = memnew(JSONRPC);

	// Set up message callback - will be set after bind_methods

	// Default server info
	server_name = "Godot MCP Server";
	server_version = "1.0.0";

	// Default capabilities - include optional listChanged fields
	server_capabilities = Dictionary();
	Dictionary tools_cap = Dictionary();
	tools_cap["listChanged"] = true;  // Server may notify when tool list changes
	server_capabilities["tools"] = tools_cap;

	Dictionary resources_cap = Dictionary();
	resources_cap["listChanged"] = true;  // Server may notify when resource list changes
	resources_cap["subscribe"] = false;    // Server does not support resource subscriptions
	server_capabilities["resources"] = resources_cap;

	Dictionary prompts_cap = Dictionary();
	prompts_cap["listChanged"] = true;  // Server may notify when prompt list changes
	server_capabilities["prompts"] = prompts_cap;

	// Set up auto-start check for next frame (after engine is initialized)
	call_deferred("check_auto_start");
}

MCPServer::~MCPServer() {
	stop_server();
	if (jsonrpc) {
		memdelete(jsonrpc);
	}
}

void MCPServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start_server", "port"), &MCPServer::start_server);
	ClassDB::bind_method(D_METHOD("stop_server"), &MCPServer::stop_server);
	ClassDB::bind_method(D_METHOD("is_server_running"), &MCPServer::is_server_running);
	ClassDB::bind_method(D_METHOD("register_tool", "name", "description_or_handler", "handler", "schema"), &MCPServer::register_tool, DEFVAL(Callable()), DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("register_resource", "name", "handler"), &MCPServer::register_resource);
	ClassDB::bind_method(D_METHOD("register_prompt", "name", "definition"), &MCPServer::register_prompt);
	ClassDB::bind_method(D_METHOD("send_response", "client_id", "request_id", "result"), &MCPServer::send_response);
	ClassDB::bind_method(D_METHOD("send_notification", "client_id", "method", "params"), &MCPServer::send_notification);
	ClassDB::bind_method(D_METHOD("send_progress", "client_id", "progress_token", "progress"), &MCPServer::send_progress);
	ClassDB::bind_method(D_METHOD("send_202_accepted", "client_id"), &MCPServer::send_202_accepted);
	ClassDB::bind_method(D_METHOD("poll"), &MCPServer::poll);
	ClassDB::bind_method(D_METHOD("_handle_message"), &MCPServer::_handle_message);
	ClassDB::bind_method(D_METHOD("_handle_server_info_resource", "params", "request_id", "client_id"), &MCPServer::_handle_server_info_resource);
	ClassDB::bind_method(D_METHOD("check_auto_start"), &MCPServer::check_auto_start);
}

void MCPServer::start_server(int p_port) {
	print_line("MCP: Attempting to start server on port ", p_port);
	if (transport.is_valid()) {
		// Set up message callback
		transport->message_callback = Callable(this, "_handle_message");

		Error err = transport->start(p_port);
		if (err != OK) {
			ERR_PRINT("Failed to start MCP server on port " + itos(p_port));
			return;
		}
		print_line("MCP: Server started successfully on port ", p_port);
	}
}

void MCPServer::stop_server() {
	if (transport.is_valid()) {
		transport->stop();
	}
}

bool MCPServer::is_server_running() const {
	if (transport.is_valid()) {
		return transport->is_listening();
	}
	return false;
}

void MCPServer::register_tool(const String &p_name, const Variant &p_description_or_handler, const Callable &p_handler, const Dictionary &p_schema) {
	if (p_handler.is_valid()) {
		// Three arguments: name, description, handler
		tool_handlers[p_name] = p_handler;
		tool_descriptions[p_name] = String(p_description_or_handler);
		tool_schemas[p_name] = p_schema;
	} else {
		// Two arguments: name, handler
		tool_handlers[p_name] = Callable(p_description_or_handler);
		tool_descriptions[p_name] = "Tool: " + p_name; // Default description for backward compatibility
		tool_schemas[p_name] = p_schema;
	}
}

void MCPServer::register_resource(const String &p_name, const Callable &p_handler) {
	resource_handlers[p_name] = p_handler;
}

void MCPServer::register_prompt(const String &p_name, const Dictionary &p_definition) {
	prompt_definitions[p_name] = p_definition;
}

void MCPServer::send_response(int64_t p_client_id, const Variant &p_request_id, const Variant &p_result) {
	Dictionary response = MCPProtocol::make_success_response(p_result, p_request_id);
	transport->send_response(p_client_id, response);
	request_to_client.erase(p_request_id);
}

void MCPServer::send_notification(int64_t p_client_id, const String &p_method, const Dictionary &p_params) {
	Dictionary notification = MCPProtocol::make_notification(p_method, p_params);
	transport->send_notification(p_client_id, notification);
}

void MCPServer::send_202_accepted(int64_t p_client_id) {
	transport->send_202_accepted(p_client_id);
}

void MCPServer::send_progress(int64_t p_client_id, const String &p_progress_token, const Dictionary &p_progress) {
	Dictionary notification = MCPProtocol::make_progress_notification(p_progress_token, p_progress);
	transport->send_notification(p_client_id, notification);
}

void MCPServer::poll() {
	if (transport.is_valid()) {
		transport->poll();
	}
}


void MCPServer::_handle_message(int64_t p_client_id, const Variant &p_message) {
	// Check if this is a batch request (Array)
	if (p_message.get_type() == Variant::ARRAY) {
		Array batch_messages = p_message;
		Array batch_results = MCPProtocol::process_batch(jsonrpc, batch_messages);
		// Send batch results back
		for (int i = 0; i < batch_results.size(); i++) {
			Variant result = batch_results[i];
			if (result.get_type() == Variant::DICTIONARY) {
				Dictionary result_dict = result;
				if (result_dict.has("id")) {
					transport->send_response(p_client_id, result_dict);
				} else {
					transport->send_notification(p_client_id, result_dict);
				}
			}
		}
		return;
	}

	// Single message (Dictionary)
	if (p_message.get_type() != Variant::DICTIONARY) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::INVALID_REQUEST,
				"Message must be Dictionary or Array",
				Variant());
		transport->send_response(p_client_id, error);
		return;
	}

	Dictionary message = p_message;
	if (!MCPProtocol::validate_message(message)) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::INVALID_REQUEST,
				"Invalid JSON-RPC message",
				MCPProtocol::get_request_id(p_message));
		transport->send_response(p_client_id, error);
		return;
	}

	String method = MCPProtocol::get_method(message);
	Variant request_id = MCPProtocol::get_request_id(message);
	Variant params = MCPProtocol::get_params(message);

	// Track request to client mapping for send_response
	if (MCPProtocol::is_request(message) && request_id.get_type() != Variant::NIL) {
		request_to_client[request_id] = p_client_id;
	}

	// Enforce initialization sequence
	if (method != "initialize" && !is_initialized) {
		// Special case: notifications/initialized is what sets the state
		if (method != "notifications/initialized") {
			if (MCPProtocol::is_request(message)) {
				Dictionary error = MCPProtocol::make_error_response(
						-32002, // Server not initialized
						"Server must be initialized first",
						request_id);
				transport->send_response(p_client_id, error);
			}
			return;
		}
	}

	if (method == "initialize") {
		Dictionary params_dict = params;
		_handle_initialize(p_client_id, params_dict, request_id);
	} else if (method == "notifications/initialized") {
		// Client has finished initialization
		is_initialized = true;
		// Per MCP spec: notifications MUST return HTTP 202 Accepted
		send_202_accepted(p_client_id);
	} else if (method == "tools/list") {
		_handle_tools_list(p_client_id, request_id);
	} else if (method == "tools/call") {
		Dictionary params_dict = params;
		_handle_tools_call(p_client_id, params_dict, request_id);
	} else if (method == "resources/list") {
		_handle_resources_list(p_client_id, request_id);
	} else if (method == "resources/read") {
		Dictionary params_dict = params;
		_handle_resources_read(p_client_id, params_dict, request_id);
	} else if (method == "prompts/list") {
		_handle_prompts_list(p_client_id, request_id);
	} else if (method == "prompts/get") {
		Dictionary params_dict = params;
		_handle_prompts_get(p_client_id, params_dict, request_id);
	} else {
		// Unknown method
		if (MCPProtocol::is_request(message)) {
			Dictionary error = MCPProtocol::make_error_response(
					JSONRPC::METHOD_NOT_FOUND,
					"Method not found: " + method,
					request_id);
			transport->send_response(p_client_id, error);
		}
		// Notifications are silently ignored if method not found
	}
}

void MCPServer::_handle_initialize(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id) {
	Dictionary result;
	
	// Always use the latest stable version
	result["protocolVersion"] = MCPProtocol::VERSION_2024_11_05;

	result["capabilities"] = server_capabilities;

	Dictionary server_info;
	server_info["name"] = server_name;
	server_info["version"] = server_version;
	result["serverInfo"] = server_info;

	// Add instructions for the model
	result["instructions"] = "This server provides tools to interact with the Godot Engine. "
							 "You can execute SafeGDScript code, manipulate the scene tree, and export scenes. "
							 "Always use 'safegdscript' as the language for script execution.";

	Dictionary response = MCPProtocol::make_success_response(result, p_request_id);
	transport->send_response(p_client_id, response);
}

void MCPServer::_handle_tools_list(int64_t p_client_id, const Variant &p_request_id) {
	Array tools;
	for (const KeyValue<String, Callable> &E : tool_handlers) {
		Dictionary tool;
		tool["name"] = E.key;
		tool["description"] = tool_descriptions.has(E.key) ? tool_descriptions[E.key] : "Tool: " + E.key;

		// Add human-readable title for UI display
		if (E.key == "execute_script") {
			tool["title"] = "Execute GDScript Code";
		} else {
			// Convert snake_case to Title Case for other tools
			String title = E.key.replace("_", " ");
			title = title.capitalize();
			tool["title"] = title;
		}

		// Use stored schema if available, otherwise provide a basic one
		Dictionary input_schema;
		if (tool_schemas.has(E.key) && !tool_schemas[E.key].is_empty()) {
			input_schema = tool_schemas[E.key];
		} else {
			input_schema["type"] = "object";
			input_schema["$schema"] = "http://json-schema.org/draft-07/schema#";
			Dictionary properties;
			input_schema["properties"] = properties;
			input_schema["additionalProperties"] = true; // Allow any properties
		}

		tool["inputSchema"] = input_schema;
		tools.push_back(tool);
	}

	Dictionary result;
	result["tools"] = tools;
	Dictionary response = MCPProtocol::make_success_response(result, p_request_id);
	transport->send_response(p_client_id, response);
}

void MCPServer::_handle_tools_call(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id) {
	if (!p_params.has("name")) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::INVALID_PARAMS,
				"Missing 'name' parameter",
				p_request_id);
		transport->send_response(p_client_id, error);
		return;
	}

	String tool_name = p_params["name"];
	if (!tool_handlers.has(tool_name)) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::METHOD_NOT_FOUND,
				"Tool not found: " + tool_name,
				p_request_id);
		transport->send_response(p_client_id, error);
		return;
	}

	Callable handler = tool_handlers[tool_name];
	Variant handler_params = p_params.has("arguments") ? p_params["arguments"] : Variant();

	// Call handler with (params, request_id, client_id) for request/response pattern
	// Handler can call send_response(client_id, request_id, result) if needed
	Variant result = handler.call(handler_params, p_request_id, p_client_id);

	// If handler returned a result, wrap it in MCP CallToolResult format
	if (result.get_type() != Variant::NIL) {
		// MCP spec requires tool results to have a 'content' field with ContentBlock array
		Dictionary tool_result;
		Array content_array;
		bool is_error = false;

		if (result.get_type() == Variant::DICTIONARY) {
			Dictionary res_dict = result;
			if (res_dict.has("success") && !(bool)res_dict["success"]) {
				is_error = true;
			}

			// If it's a dictionary, we can try to extract meaningful content
			String text_content;
			if (res_dict.has("message")) {
				text_content += String(res_dict["message"]) + "\n";
			}
			
			JSON json;
			if (res_dict.has("result")) {
				text_content += "Result: " + json.stringify(res_dict["result"]) + "\n";
			}
			
			if (res_dict.has("results")) {
				text_content += "Action Results: " + json.stringify(res_dict["results"]) + "\n";
			}

			if (res_dict.has("exported_results")) {
				text_content += "Exported Results: " + json.stringify(res_dict["exported_results"]) + "\n";
			}

			if (res_dict.has("error")) {
				text_content += "Error: " + String(res_dict["error"]) + "\n";
			}

			if (text_content.is_empty()) {
				// Fallback to stringifying the whole dictionary if no specific fields found
				text_content = json.stringify(res_dict);
			}

			Dictionary content_block;
			content_block["type"] = "text";
			content_block["text"] = text_content.strip_edges();
			content_array.push_back(content_block);
		} else if (result.get_type() == Variant::STRING) {
			Dictionary content_block;
			content_block["type"] = "text";
			content_block["text"] = result;
			content_array.push_back(content_block);
		} else {
			JSON json;
			Dictionary content_block;
			content_block["type"] = "text";
			content_block["text"] = json.stringify(result);
			content_array.push_back(content_block);
		}

		tool_result["content"] = content_array;
		tool_result["isError"] = is_error;

		// Also include structured content for backwards compatibility
		tool_result["structuredContent"] = result;
		send_response(p_client_id, p_request_id, tool_result);
	}
	// If handler didn't return (async pattern), it should call send_response itself
}

void MCPServer::_handle_resources_list(int64_t p_client_id, const Variant &p_request_id) {
	Array resources;
	for (const KeyValue<String, Callable> &E : resource_handlers) {
		Dictionary resource;
		resource["uri"] = "resource://" + E.key;
		resource["name"] = E.key;
		resource["description"] = "Resource: " + E.key;
		resource["mimeType"] = "application/json";
		resources.push_back(resource);
	}

	Dictionary result;
	result["resources"] = resources;
	Dictionary response = MCPProtocol::make_success_response(result, p_request_id);
	transport->send_response(p_client_id, response);
}

void MCPServer::_handle_resources_read(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id) {
	if (!p_params.has("uri")) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::INVALID_PARAMS,
				"Missing 'uri' parameter",
				p_request_id);
		transport->send_response(p_client_id, error);
		return;
	}

	String uri = p_params["uri"];
	String resource_name = uri.replace("resource://", "");

	if (!resource_handlers.has(resource_name)) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::METHOD_NOT_FOUND,
				"Resource not found: " + resource_name,
				p_request_id);
		transport->send_response(p_client_id, error);
		return;
	}

	Callable handler = resource_handlers[resource_name];
	Dictionary handler_params;
	handler_params["uri"] = uri;

	Variant result = handler.call(handler_params, p_request_id, p_client_id);

	if (result.get_type() != Variant::NIL) {
		Dictionary final_result;
		Array contents;

		if (result.get_type() == Variant::DICTIONARY) {
			Dictionary res_dict = result;
			if (res_dict.has("contents")) {
				final_result = res_dict;
			} else {
				Dictionary content;
				content["uri"] = uri;
				content["text"] = JSON::stringify(res_dict);
				contents.push_back(content);
				final_result["contents"] = contents;
			}
		} else if (result.get_type() == Variant::STRING) {
			Dictionary content;
			content["uri"] = uri;
			content["text"] = result;
			contents.push_back(content);
			final_result["contents"] = contents;
		} else {
			Dictionary content;
			content["uri"] = uri;
			content["text"] = JSON::stringify(result);
			contents.push_back(content);
			final_result["contents"] = contents;
		}
		send_response(p_client_id, p_request_id, final_result);
	}
}

void MCPServer::_handle_prompts_list(int64_t p_client_id, const Variant &p_request_id) {
	Array prompts;
	for (const KeyValue<String, Dictionary> &E : prompt_definitions) {
		Dictionary prompt_info;
		prompt_info["name"] = E.key;

		Dictionary definition = E.value;
		if (definition.has("title")) {
			prompt_info["title"] = definition["title"];
		}
		if (definition.has("description")) {
			prompt_info["description"] = definition["description"];
		}
		if (definition.has("arguments")) {
			prompt_info["arguments"] = definition["arguments"];
		}

		prompts.push_back(prompt_info);
	}

	Dictionary result;
	result["prompts"] = prompts;
	Dictionary response = MCPProtocol::make_success_response(result, p_request_id);
	transport->send_response(p_client_id, response);
}

void MCPServer::_handle_prompts_get(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id) {
	if (!p_params.has("name")) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::INVALID_PARAMS,
				"Missing 'name' parameter",
				p_request_id);
		transport->send_response(p_client_id, error);
		return;
	}

	String prompt_name = p_params["name"];
	if (!prompt_definitions.has(prompt_name)) {
		Dictionary error = MCPProtocol::make_error_response(
				JSONRPC::METHOD_NOT_FOUND,
				"Prompt not found: " + prompt_name,
				p_request_id);
		transport->send_response(p_client_id, error);
		return;
	}

	Dictionary definition = prompt_definitions[prompt_name];
	Dictionary result;

	if (definition.has("description")) {
		result["description"] = definition["description"];
	}

	if (definition.has("messages")) {
		result["messages"] = definition["messages"];
	}

	Dictionary response = MCPProtocol::make_success_response(result, p_request_id);
	transport->send_response(p_client_id, response);
}

#include "mcp_runner.h"
#include "scene/main/window.h"
#include "scene/main/scene_tree.h"

void MCPServer::check_auto_start() {
	if (should_auto_start() && !is_server_running()) {
		start_server(auto_start_port);
		if (is_server_running()) {
			// Add runner to scene tree to handle polling
			MainLoop *main_loop = OS::get_singleton()->get_main_loop();
			SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);
			if (scene_tree) {
				MCPRunner *runner = memnew(MCPRunner);
				scene_tree->get_root()->add_child(runner);
				print_line("MCP: Added runner to scene tree");
			}

			// Register tools from MCPScriptServer
			MCPScriptServer *script_server = Object::cast_to<MCPScriptServer>(Engine::get_singleton()->get_singleton_object("MCPScriptServer"));
			if (script_server) {
				// Register compile_script tool
				String compile_description = R"(
Validate that SafeGDScript code compiles without executing it.

This tool checks syntax and compilation errors without running the code. Useful for validating scripts before execution.

Examples of SafeGDScript:

Basic syntax check:
@export var value: int = 42

func my_method():
    return value

Parameters: code (string), language (must be 'safegdscript')
)";

				Dictionary compile_schema;
				compile_schema["type"] = "object";
				Dictionary compile_properties;
				Dictionary code_prop;
				code_prop["type"] = "string";
				code_prop["description"] = "The SafeGDScript code to validate";
				compile_properties["code"] = code_prop;
				Dictionary language_prop;
				language_prop["type"] = "string";
				Array language_enum;
				language_enum.push_back("safegdscript");
				language_prop["enum"] = language_enum;
				language_prop["default"] = "safegdscript";
				language_prop["description"] = "Language type (must be 'safegdscript')";
				compile_properties["language"] = language_prop;
				compile_schema["properties"] = compile_properties;
				Array compile_required;
				compile_required.push_back("code");
				compile_schema["required"] = compile_required;

				Callable compile_handler = Callable(script_server, "_tool_compile_script");
				register_tool("compile_script", compile_description, compile_handler, compile_schema);

				// Register execute_script tool
				String execute_description = R"(
Execute a complete GDScript (SafeGDScript subset) with security restrictions.
The script can contain methods which can be called and their results returned.

Example:
func my_method():
    return 2 + 3

Parameters:
- code (string): The SafeGDScript code to execute.
- language (string): Must be 'safegdscript'.
- method (string, optional): The method to call in the script (e.g., 'run'). If not provided, the script is initialized but no method is called.
)";

				Dictionary execute_schema;
				execute_schema["type"] = "object";
				Dictionary execute_properties;
				Dictionary code_prop_exec;
				code_prop_exec["type"] = "string";
				code_prop_exec["description"] = "The SafeGDScript code to execute";
				execute_properties["code"] = code_prop_exec;
				Dictionary language_prop_exec;
				language_prop_exec["type"] = "string";
				Array language_enum_exec;
				language_enum_exec.push_back("safegdscript");
				language_prop_exec["enum"] = language_enum_exec;
				language_prop_exec["default"] = "safegdscript";
				language_prop_exec["description"] = "Language type (must be 'safegdscript')";
				execute_properties["language"] = language_prop_exec;
				Dictionary method_prop_exec;
				method_prop_exec["type"] = "string";
				method_prop_exec["description"] = "The method to call in the script";
				execute_properties["method"] = method_prop_exec;
				execute_schema["properties"] = execute_properties;
				Array execute_required;
				execute_required.push_back("code");
				execute_schema["required"] = execute_required;

				Callable execute_handler = Callable(script_server, "_tool_execute_script");
				register_tool("execute_script", execute_description, execute_handler, execute_schema);

				// Register execute_action_sequence tool
				String executor_description = R"(
Execute a sequence of actions on the Godot scene tree with timeout protection.

This is a simple forward executor that runs predefined sequences of scene tree operations.
Actions are executed in order and can include generic method calls and SafeGDScript execution.
All actions use unified security restrictions to prevent dangerous operations.
Execution is limited to 5 seconds by default to prevent infinite loops.

Available actions:
- command_call: Generic method call with unified security (create_node, set_property, etc.)
- command_set_script: Execute SafeGDScript code.
  Parameters:
  - script (string): The SafeGDScript code.
  - target_path (string, optional): Path to the node to set the script on.
  - method (string, optional): The method to call (e.g., 'run').
  - input_properties (object, optional): Properties to set before execution.

Parameters: actions (array of action objects), timeout (number, optional, default 5.0)
)";

				Dictionary executor_schema;
				executor_schema["type"] = "object";
				Dictionary executor_properties;
				Dictionary actions_prop;
				actions_prop["type"] = "array";
				actions_prop["description"] = "Array of action objects to execute";
				executor_properties["actions"] = actions_prop;
				Dictionary timeout_prop;
				timeout_prop["type"] = "number";
				timeout_prop["default"] = 5.0;
				timeout_prop["description"] = "Maximum execution time in seconds";
				executor_properties["timeout"] = timeout_prop;
				executor_schema["properties"] = executor_properties;
				Array executor_required;
				executor_required.push_back("actions");
				executor_schema["required"] = executor_required;

				Callable executor_handler = Callable(script_server, "_tool_execute_action_sequence");
				register_tool("execute_action_sequence", executor_description, executor_handler, executor_schema);

				// Register save_mesh_to_escn prompt
				Dictionary save_mesh_prompt;
				save_mesh_prompt["title"] = "Save Mesh to ESCN";
				save_mesh_prompt["description"] = "Step-by-step guide for creating a MeshInstance3D, adding a primitive mesh via script, and exporting to ESCN format";

				Array arguments;
				Dictionary mesh_type_arg;
				mesh_type_arg["name"] = "mesh_type";
				mesh_type_arg["description"] = "Type of primitive mesh (BoxMesh, SphereMesh, CylinderMesh, etc.)";
				mesh_type_arg["required"] = false;
				arguments.push_back(mesh_type_arg);

				Dictionary output_path_arg;
				output_path_arg["name"] = "output_path";
				output_path_arg["description"] = "Output path for the ESCN file (optional)";
				output_path_arg["required"] = false;
				arguments.push_back(output_path_arg);

				save_mesh_prompt["arguments"] = arguments;

				Array messages;
				Dictionary message;
				message["role"] = "user";

				Dictionary content;
				content["type"] = "text";
				content["text"] = R"PROMPT(
To save a mesh to ESCN format in Godot:

1. **Create a MeshInstance3D node**:
   Use the `execute_action_sequence` tool with `command_call` action:
   ```json
   {
     "actions": [{
       "action": "command_call",
       "method": "create_node",
       "node_type": "MeshInstance3D",
       "name": "MyMesh",
       "parent_path": "/root"
     }]
   }
   ```

2. **Add a script to create and set a primitive mesh**:
   Use `command_set_script` to attach SafeGDScript that initializes the mesh in `_ready()`:
   ```json
   {
     "actions": [{
       "action": "command_set_script",
       "target_path": "/root/MyMesh",
       "script": "func _ready():\n    var box = BoxMesh.new()\n    box.size = Vector3(2.0, 2.0, 2.0)\n    mesh = box"
     }]
   }
   ```
   The script will automatically execute `_ready()` when set, which creates the BoxMesh and assigns it directly to the MeshInstance3D's `mesh` property.

3. **Export the scene as ESCN**:
   Use the `command_export_scene` action to save the scene in text format:
   ```json
   {
     "actions": [{
       "action": "command_export_scene",
       "node_path": "/root/MyMesh",
       "format": "escn"
     }]
   }
   ```

4. **Get the exported scene data**:
   The `command_export_scene` action returns the scene data as plain text (not base64) with MIME type `text/plain`.
   The exported scene file is in Godot's text-based ESCN format and can be saved directly to a file.

**Available primitive meshes**: BoxMesh, SphereMesh, CylinderMesh, CapsuleMesh, PlaneMesh, PrismMesh, TorusMesh

**Example complete workflow**:
```json
{
  "actions": [
    {
      "action": "command_call",
      "method": "create_node",
      "node_type": "MeshInstance3D",
      "name": "PrimitiveMesh",
      "parent_path": "/root"
    },
    {
      "action": "command_set_script",
      "target_path": "/root/PrimitiveMesh",
      "script": "func _ready():\n    var sphere = SphereMesh.new()\n    sphere.radius = 1.0\n    sphere.height = 2.0\n    mesh = sphere"
    },
    {
      "action": "command_export_scene",
      "node_path": "/root/PrimitiveMesh",
      "format": "escn"
    }
  ]
}
```

**Notes**:
- Use `target_path` (not `node_path`) for `command_set_script` to specify which node to attach the script to
- The script uses `_ready()` which is automatically called when the script is set on the node
- Set `mesh` directly in the script (not `@export var mesh`) - this sets the MeshInstance3D's mesh property directly
- The exported scene data is returned as plain text, ready to save as a `.escn` file
)PROMPT";

				message["content"] = content;
				messages.push_back(message);
				save_mesh_prompt["messages"] = messages;

				register_prompt("save_mesh_to_escn", save_mesh_prompt);

				// Register a test resource
				register_resource("server_info", Callable(this, "_handle_server_info_resource"));
			} else {
				print_line("MCP: Warning - MCPScriptServer singleton not found, tools not registered");
			}
		} else {
			print_line("MCP: Failed to auto-start server on port ", auto_start_port);
		}
	}
}

Variant MCPServer::_handle_server_info_resource(const Dictionary &p_params, const Variant &p_request_id, int64_t p_client_id) {
	Dictionary info;
	info["name"] = server_name;
	info["version"] = server_version;
	info["status"] = "running";
	return info;
}

bool MCPServer::should_auto_start() const {
	// Check for port environment variable first
	String port_env = OS::get_singleton()->get_environment("GODOT_MCP_SERVER_PORT");
	if (!port_env.is_empty()) {
		auto_start_port = port_env.to_int();
	}

	// Auto-start if GODOT_MCP_SERVER_START environment variable is set (from --mcp-server CLI flag)
	String mcp_server_env = OS::get_singleton()->get_environment("GODOT_MCP_SERVER_START");
	if (!mcp_server_env.is_empty() && mcp_server_env.to_lower() == "true") {
		return true;
	}

	return false;
}
