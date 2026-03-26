/**************************************************************************/
/*  scenetree_mcp.cpp                                                     */
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

#include "scenetree_mcp.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/ip_address.h"
#include "core/io/json.h"
#include "core/object/class_db.h"

static void _send_raw_string(const Ref<StreamPeerTCP> &p_peer, const String &p_text) {
	if (!p_peer.is_valid()) {
		return;
	}
	CharString utf8 = p_text.utf8();
	p_peer->put_data((const uint8_t *)utf8.get_data(), utf8.length());
}

void SceneTreeMCP::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start_server", "port"), &SceneTreeMCP::start_server, DEFVAL(8765));
	ClassDB::bind_method(D_METHOD("stop_server"), &SceneTreeMCP::stop_server);
	ClassDB::bind_method(D_METHOD("is_server_running"), &SceneTreeMCP::is_server_running);
	ClassDB::bind_method(D_METHOD("start_http_server", "port"), &SceneTreeMCP::start_http_server, DEFVAL(8765));
	ClassDB::bind_method(D_METHOD("stop_http_server"), &SceneTreeMCP::stop_http_server);
	ClassDB::bind_method(D_METHOD("is_http_server_running"), &SceneTreeMCP::is_http_server_running);
	ClassDB::bind_method(D_METHOD("register_tool", "name", "description", "schema", "callable"),
			&SceneTreeMCP::register_tool);
	ClassDB::bind_method(D_METHOD("register_resource", "uri", "name", "callable"),
			&SceneTreeMCP::register_resource);
	ClassDB::bind_method(D_METHOD("set_server_info", "name", "version"), &SceneTreeMCP::set_server_info);
	ClassDB::bind_method(D_METHOD("get_server_name"), &SceneTreeMCP::get_server_name);
	ClassDB::bind_method(D_METHOD("get_server_version"), &SceneTreeMCP::get_server_version);
	ClassDB::bind_method(D_METHOD("process_server", "delta"), &SceneTreeMCP::process_server);
	ClassDB::bind_method(D_METHOD("_read_project_filesystem_resource"), &SceneTreeMCP::_read_project_filesystem_resource);

	// ClassDB method bindings
	ClassDB::bind_method(D_METHOD("classdb_instantiate", "args"), &SceneTreeMCP::classdb_instantiate);
	ClassDB::bind_method(D_METHOD("classdb_can_instantiate", "args"), &SceneTreeMCP::classdb_can_instantiate);
	ClassDB::bind_method(D_METHOD("classdb_get_method_list", "args"), &SceneTreeMCP::classdb_get_method_list);
	ClassDB::bind_method(D_METHOD("classdb_class_exists", "args"), &SceneTreeMCP::classdb_class_exists);
}

SceneTreeMCP::SceneTreeMCP() {
	_register_builtin_method_tools();
	_register_builtin_prompts();
	register_resource("res://", "Godot project filesystem (non-binary files)", Callable(this, "_read_project_filesystem_resource"));
}

SceneTreeMCP::~SceneTreeMCP() {
	// _finalize() will be called automatically if used as main loop
}

Error SceneTreeMCP::start_server(int p_port) {
	if (server_running) {
		return ERR_ALREADY_IN_USE;
	}
	_start_transport(p_port);
	return OK;
}

void SceneTreeMCP::stop_server() {
	_stop_transport();
}

Error SceneTreeMCP::start_http_server(int p_port) {
	if (http_server_running) {
		return ERR_ALREADY_IN_USE;
	}
	http_server.instantiate();
	IPAddress bind_addr("127.0.0.1");
	Error err = http_server->listen(p_port, bind_addr);
	if (err != OK) {
		http_server.unref();
		http_server_running = false;
		return err;
	}
	http_server_port = p_port;
	http_server_running = true;
	is_initialized = false;
	return OK;
}

void SceneTreeMCP::stop_http_server() {
	if (!http_server_running) {
		return;
	}
	http_server.unref();
	http_server_running = false;
}

void SceneTreeMCP::process_server(double p_delta) {
	if (server_running || http_server_running) {
		_accept_connections();
	}
	if (server_running || http_server_running) {
		_process_peers();
		_prune_dead_peers();
	}
}

// MainLoop lifecycle - automatically called by Godot when this is the active main loop
void SceneTreeMCP::initialize() {
	// Auto-start MCP server on port 8765
	_start_transport(8765);
}

bool SceneTreeMCP::process(double p_delta) {
	// Auto-process MCP server every frame
	if (server_running || http_server_running) {
		_accept_connections();
		_process_peers();
		_prune_dead_peers();
	}
	// Call parent process to maintain SceneTree functionality
	return SceneTree::process(p_delta);
}

void SceneTreeMCP::finalize() {
	_stop_transport();
}

void SceneTreeMCP::_start_transport(int p_port) {
	tcp_server.instantiate();
	IPAddress bind_addr("127.0.0.1");
	Error err = tcp_server->listen(p_port, bind_addr);
	if (err != OK) {
		server_running = false;
		return;
	}

	server_port = p_port;
	server_running = true;
	is_initialized = false;
}

void SceneTreeMCP::_stop_transport() {
	if (!server_running) {
		return;
	}

	peers.clear();
	tcp_server.unref();
	server_running = false;
}

void SceneTreeMCP::_accept_connections() {
	// Accept connections from TCP server (backward compatibility)
	if (server_running && tcp_server.is_valid()) {
		while (tcp_server->is_connection_available()) {
			Ref<StreamPeerTCP> peer = tcp_server->take_connection();
			if (peer.is_valid() && peer->get_status() != StreamPeerTCP::STATUS_NONE) {
				MCPServerPeer new_peer;
				new_peer.peer = peer;
				new_peer.is_http = false;
				peers.push_back(new_peer);
			} else {
				break;
			}
		}
	}

	// Accept connections from HTTP server
	if (http_server_running && http_server.is_valid()) {
		while (http_server->is_connection_available()) {
			Ref<StreamPeerTCP> peer = http_server->take_connection();
			if (peer.is_valid() && peer->get_status() != StreamPeerTCP::STATUS_NONE) {
				MCPServerPeer new_peer;
				new_peer.peer = peer;
				new_peer.is_http = true;
				peers.push_back(new_peer);
			} else {
				break;
			}
		}
	}
}

bool SceneTreeMCP::_is_http_peer(const MCPServerPeer &p_peer) {
	return p_peer.is_http;
}

void SceneTreeMCP::_process_peers() {
	for (int64_t i = 0; i < peers.size(); i++) {
		MCPServerPeer &peer = peers[i];
		if (!peer.peer.is_valid()) {
			continue;
		}

		peer.peer->poll();

		int status = peer.peer->get_status();
		// Skip disconnected or errored peers
		if (status != StreamPeerTCP::STATUS_CONNECTED) {
			continue;
		}

		int available = peer.peer->get_available_bytes();
		if (available > 0) {
			String data = peer.peer->get_utf8_string(available);
			peer.request_buffer += data;

			if (!_is_http_peer(peer)) {
				JSON json;
				Error parse_err = json.parse(peer.request_buffer);
				if (parse_err != OK) {
					continue;
				}

				Variant payload = json.get_data();
				bool should_reply = false;
				Variant response = _handle_jsonrpc_request(payload, should_reply);

				if (should_reply && response.get_type() != Variant::NIL) {
					String response_json = json.stringify(response);
					_send_raw_string(peer.peer, response_json);
				}

				peer.request_buffer = "";
				continue;
			}

			int body_start = peer.request_buffer.find("\r\n\r\n");
			if (body_start == -1) {
				continue;
			}

			String headers_section = peer.request_buffer.substr(0, body_start);
			String body = peer.request_buffer.substr(body_start + 4);

			String method, path;
			Dictionary headers;
			(void)_parse_http_request(headers_section, method, path, headers);

			if (path != "/mcp") {
				String error_response = _format_http_response(404, "Not Found",
						"{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32600,\"message\":\"Not Found\"}}");
				_send_raw_string(peer.peer, error_response);
				peer.request_buffer = "";
				continue;
			}

			if (!body.is_empty()) {
				JSON json;
				Error parse_err = json.parse(body);
				if (parse_err == OK) {
					Variant payload = json.get_data();
					bool should_reply = false;
					Variant response = _handle_jsonrpc_request(payload, should_reply);

					if (should_reply && response.get_type() != Variant::NIL) {
						String response_json = json.stringify(response);
						String http_response = _format_http_response(200, "OK", response_json, headers);
						_send_raw_string(peer.peer, http_response);
					}
				} else {
					Dictionary parse_error = _make_jsonrpc_error(-32700, "Parse error", Variant());
					JSON json2;
					String error_json = json2.stringify(parse_error);
					String http_response = _format_http_response(400, "Bad Request", error_json);
					_send_raw_string(peer.peer, http_response);
				}
			}

			peer.request_buffer = "";
		}
	}
}

void SceneTreeMCP::_prune_dead_peers() {
	LocalVector<MCPServerPeer> alive;
	alive.reserve(peers.size());
	for (int64_t i = 0; i < peers.size(); i++) {
		const MCPServerPeer &peer = peers[i];
		if (peer.peer.is_valid() && peer.peer->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
			alive.push_back(peer);
		}
	}
	peers = alive;
}

String SceneTreeMCP::_parse_http_request(const String &p_data, String &r_method,
		String &r_path, Dictionary &r_headers) {
	Vector<String> lines_vec = p_data.split("\r\n");
	Array lines;
	for (const String &line : lines_vec) {
		lines.push_back(line);
	}
	if (lines.is_empty()) {
		return "";
	}

	String request_line = lines[0];
	Vector<String> parts_vec = request_line.split(" ");
	Array parts;
	for (const String &part : parts_vec) {
		parts.push_back(part);
	}
	if (parts.size() >= 2) {
		r_method = parts[0];
		r_path = parts[1];
	}

	for (int i = 1; i < lines.size(); i++) {
		String line = lines[i];
		int colon_pos = line.find(":");
		if (colon_pos != -1) {
			String name = line.substr(0, colon_pos).strip_edges();
			String value = line.substr(colon_pos + 1).strip_edges();
			r_headers[name.to_lower()] = value;
		}
	}

	return r_method;
}

String SceneTreeMCP::_format_http_response(int p_status_code, const String &p_status_text,
		const String &p_body, const Dictionary &p_headers) {
	// Determine if this is a streaming response
	bool is_streaming = _is_streaming_request(p_headers);
	String content_type = _get_content_type(p_headers);

	String response = "HTTP/1.1 ";
	response += itos(p_status_code);
	response += " ";
	response += p_status_text;
	response += "\r\n";

	// Set Content-Type header
	response += "Content-Type: ";
	response += content_type;
	response += "\r\n";

	if (is_streaming) {
		// For streaming responses, we use chunked transfer encoding
		// Content-Length is not set for streaming responses
		response += "Transfer-Encoding: chunked\r\n";
	} else {
		// For non-streaming responses, use Content-Length
		response += "Content-Length: ";
		response += itos(p_body.length());
		response += "\r\n";
	}

	response += "Connection: keep-alive\r\n";

	// Add custom headers
	Array header_keys = p_headers.keys();
	for (int i = 0; i < header_keys.size(); i++) {
		String header_name = header_keys[i];
		// Skip Content-Type and Transfer-Encoding as we handle them above
		if (header_name.to_lower() != "content-type" && header_name.to_lower() != "transfer-encoding") {
			String header_value = p_headers[header_name];
			response += header_name;
			response += ": ";
			response += header_value;
			response += "\r\n";
		}
	}

	response += "\r\n";

	if (!is_streaming) {
		response += p_body;
	}

	return response;
}

Error SceneTreeMCP::register_tool(const String &p_name, const String &p_description,
		const Dictionary &p_schema, const Callable &p_callable) {
	if (p_name.is_empty() || !p_callable.is_valid()) {
		return ERR_INVALID_DATA;
	}

	RegisteredTool tool;
	tool.name = p_name;
	tool.description = p_description;
	tool.schema = p_schema;
	tool.callable = p_callable;

	tool_registry[p_name] = tool;
	return OK;
}

Error SceneTreeMCP::register_resource(const String &p_uri, const String &p_name,
		const Callable &p_callable) {
	if (p_uri.is_empty() || !p_callable.is_valid()) {
		return ERR_INVALID_DATA;
	}

	RegisteredResource resource;
	resource.uri = p_uri;
	resource.name = p_name;
	resource.callable = p_callable;

	resource_registry[p_uri] = resource;
	return OK;
}

void SceneTreeMCP::set_server_info(const String &p_name, const String &p_version) {
	server_info_name = p_name;
	server_info_version = p_version;
}

void SceneTreeMCP::_register_builtin_method_tools() {
	_register_class_methods_as_tools("Object", "Object");
	_register_class_methods_as_tools("MainLoop", "MainLoop");
	_register_class_methods_as_tools("SceneTree", "SceneTree");
	_register_class_methods_as_tools("SceneTreeMCP", "SceneTreeMCP");
	_register_classdb_methods_as_tools();
}

void SceneTreeMCP::_register_builtin_prompts() {
	RegisteredPrompt prompt;
	prompt.name = "godot_mcp_usage";
	prompt.description = "How to use the Godot SceneTreeMCP server (tools, prompts, resources).";
	prompt.arguments = Array();
	prompt.template_text =
			"Use this server with JSON-RPC 2.0 over the SceneTreeMCP transport.\n"
			"1) Send initialize with protocolVersion '2025-03-26' (or '2024-11-05' for compatibility).\n"
			"2) Call tools/list, then tools/call with {name, arguments}.\n"
			"3) Call resources/list and resources/read for project data.\n"
			"4) The resource 'res://' returns a recursive list of project files, excluding binary files.\n"
			"5) Available class tools are exposed under Object.*, MainLoop.*, SceneTree.*, and SceneTreeMCP.*.";

	prompt_registry[prompt.name] = prompt;
}

void SceneTreeMCP::_register_class_methods_as_tools(const StringName &p_class_name, const String &p_prefix) {
	List<MethodInfo> methods;
	ClassDB::get_method_list(p_class_name, &methods, true);

	Dictionary schema = _build_generic_method_schema();

	for (const MethodInfo &method_info : methods) {
		if (method_info.name.is_empty()) {
			continue;
		}

		String method_name = String(method_info.name);
		String tool_name = p_prefix + "." + method_name;

		if (tool_registry.has(tool_name)) {
			continue;
		}

		Callable callable(this, method_info.name);
		if (!callable.is_valid()) {
			continue;
		}

		String description = "Invoke " + p_prefix + "::" + method_name + " on SceneTreeMCP.";
		register_tool(tool_name, description, schema, callable);
	}
}

void SceneTreeMCP::_register_classdb_methods_as_tools() {
	// Register ClassDB.instantiate as a tool for dynamic class instantiation
	{
		Dictionary schema;
		schema["type"] = "array";
		schema["items"] = Dictionary();
		schema["description"] = "Class name to instantiate (String) and optional arguments (Array)";

		Callable callable = Callable(this, "classdb_instantiate");
		if (callable.is_valid()) {
			register_tool("ClassDB.instantiate", "Instantiate a Godot class by name using ClassDB.instantiate()", schema, callable);
		}
	}

	// Register ClassDB.can_instantiate as a tool for checking if a class can be instantiated
	{
		Dictionary schema;
		schema["type"] = "array";
		schema["items"] = Dictionary();
		schema["description"] = "Class name to check (String)";

		Callable callable = Callable(this, "classdb_can_instantiate");
		if (callable.is_valid()) {
			register_tool("ClassDB.can_instantiate", "Check if a Godot class can be instantiated using ClassDB.can_instantiate()", schema, callable);
		}
	}

	// Register ClassDB.get_method_list as a tool for discovering class methods
	{
		Dictionary schema;
		schema["type"] = "array";
		schema["items"] = Dictionary();
		schema["description"] = "Class name to get methods from (String)";

		Callable callable = Callable(this, "classdb_get_method_list");
		if (callable.is_valid()) {
			register_tool("ClassDB.get_method_list", "Get list of methods for a Godot class using ClassDB.get_method_list()", schema, callable);
		}
	}

	// Register ClassDB.class_exists as a tool for checking if a class exists
	{
		Dictionary schema;
		schema["type"] = "array";
		schema["items"] = Dictionary();
		schema["description"] = "Class name to check (String)";

		Callable callable = Callable(this, "classdb_class_exists");
		if (callable.is_valid()) {
			register_tool("ClassDB.class_exists", "Check if a Godot class exists using ClassDB.class_exists()", schema, callable);
		}
	}

	// Register SceneTree.get_root as a tool for accessing the scene root node
	{
		Dictionary schema;
		schema["type"] = "array";
		schema["items"] = Dictionary();
		schema["description"] = "No parameters required - gets the root Node of the current scene";

		Callable callable = Callable(this, "scenetree_get_root");
		if (callable.is_valid()) {
			register_tool("SceneTree.get_root", "Get the root Node of the current scene using SceneTree.get_root()", schema, callable);
		}
	}
}

// ClassDB method wrappers
Variant SceneTreeMCP::classdb_instantiate(const Array &p_args) {
	if (p_args.is_empty() || p_args[0].get_type() != Variant::STRING) {
		return Variant("Error: First argument must be a class name (string)");
	}

	String class_name = p_args[0];

	// Call ClassDB::instantiate directly
	Object *obj = ClassDB::instantiate(class_name);

	if (obj) {
		// Return the object data
		Dictionary ret;
		ret["success"] = true;
		ret["class"] = class_name;
		ret["object"] = Variant(obj);
		return ret;
	}

	// Return error information
	Dictionary ret;
	ret["success"] = false;
	ret["class"] = class_name;

	// Get more specific error information
	ClassDB::ClassInfo *ci = ClassDB::classes.getptr(class_name);
	if (ci) {
		if (ci->disabled) {
			ret["error"] = "Class is disabled";
		} else if (ci->is_virtual) {
			ret["error"] = "Class is virtual and cannot be instantiated";
		} else {
			ret["error"] = "Failed to instantiate class (unknown reason)";
		}
	} else {
		ret["error"] = "Class does not exist";
	}

	return ret;
}

Variant SceneTreeMCP::scenetree_get_root(const Array &p_args) {
	// Get the current SceneTree instance
	SceneTree *scene_tree = OS::get_singleton()->get_main_loop();
	
	if (scene_tree && scene_tree->is_inside_tree()) {
		// Get the root node of the scene
		Node *root_node = scene_tree->get_root();
		
		if (root_node) {
			// Return the root node information
			Dictionary ret;
			ret["success"] = true;
			ret["root"] = Variant(root_node);
			ret["root_name"] = root_node->get_name();
			ret["root_type"] = root_node->get_class();
			return ret;
		}
	}
	
	// Return error if scene tree or root not available
	Dictionary ret;
	ret["success"] = false;
	ret["error"] = "SceneTree or root node not available";
	return ret;
}

Variant SceneTreeMCP::classdb_can_instantiate(const Array &p_args) {
	if (p_args.is_empty() || p_args[0].get_type() != Variant::STRING) {
		return false;
	}

	String class_name = p_args[0];

	// Call ClassDB::can_instantiate directly
	return ClassDB::can_instantiate(class_name);
}

Variant SceneTreeMCP::classdb_get_method_list(const Array &p_args) {
	if (p_args.is_empty() || p_args[0].get_type() != Variant::STRING) {
		return Array();
	}

	String class_name = p_args[0];
	List<MethodInfo> methods;
	ClassDB::get_method_list(class_name, &methods, true);

	Array result;
	for (const MethodInfo &method_info : methods) {
		Dictionary method;
		method["name"] = method_info.name;
		method["return_type"] = method_info.return_val.type;

		Array args;
		for (const PropertyInfo &arg_info : method_info.arguments) {
			Dictionary arg;
			arg["name"] = arg_info.name;
			arg["type"] = arg_info.type;
			arg["hint"] = arg_info.hint;
			args.push_back(arg);
		}
		method["arguments"] = args;

		result.push_back(method);
	}

	return result;
}

Variant SceneTreeMCP::classdb_class_exists(const Array &p_args) {
	if (p_args.is_empty() || p_args[0].get_type() != Variant::STRING) {
		return false;
	}

	String class_name = p_args[0];
	return ClassDB::class_exists(class_name);
}

Dictionary SceneTreeMCP::_build_generic_method_schema() const {
	Dictionary schema;
	schema["type"] = "array";
	schema["description"] = "Positional argument array passed to the bound method.";
	return schema;
}

Variant SceneTreeMCP::_handle_jsonrpc_request(const Variant &p_request, bool &r_should_reply) {
	r_should_reply = true;

	// Handle batch requests (array of requests)
	if (p_request.get_type() == Variant::ARRAY) {
		return _handle_jsonrpc_batch(p_request, r_should_reply);
	}

	if (p_request.get_type() != Variant::DICTIONARY) {
		return _make_jsonrpc_error(-32600, "Invalid Request", Variant());
	}

	// Single request handling - cast to Dictionary
	Dictionary request = p_request;

	if (!request.has("id")) {
		r_should_reply = false;
	}

	Variant id = request.has("id") ? request.get("id", Variant()) : Variant();

	if (!request.has("jsonrpc") || request.get("jsonrpc", Variant()) != "2.0") {
		return _make_jsonrpc_error(-32600, "Invalid Request", id);
	}

	if (!request.has("method")) {
		return _make_jsonrpc_error(-32600, "Invalid Request", id);
	}

	String method = request.get("method", Variant());

	if (method == "initialize") {
		return _handle_mcp_initialize(request, id);
	} else if (method == "initialized") {
		return _handle_mcp_initialized(request, id);
	} else if (method == "tools/list") {
		return _handle_mcp_tools_list(request, id);
	} else if (method == "tools/call") {
		return _handle_mcp_tools_call(request, id);
	} else if (method == "resources/list") {
		return _handle_mcp_resources_list(request, id);
	} else if (method == "resources/read") {
		return _handle_mcp_resources_read(request, id);
	} else if (method == "prompts/list") {
		return _handle_mcp_prompts_list(request, id);
	} else if (method == "prompts/get") {
		return _handle_mcp_prompts_get(request, id);
	} else if (method == "ping") {
		return _handle_mcp_ping(request, id);
	} else {
		return _make_jsonrpc_error(-32601, "Method not found: " + method, id);
	}
}

Variant SceneTreeMCP::_handle_jsonrpc_batch(const Array &p_batch, bool &r_should_reply) {
	r_should_reply = true;

	if (p_batch.is_empty()) {
		r_should_reply = true;
		Array invalid_batch;
		invalid_batch.push_back(_make_jsonrpc_error(-32600, "Invalid Request", Variant()));
		return invalid_batch;
	}

	Array responses;
	bool any_reply = false;

	for (int i = 0; i < p_batch.size(); i++) {
		Variant entry = p_batch[i];
		if (entry.get_type() != Variant::DICTIONARY) {
			any_reply = true;
			responses.push_back(_make_jsonrpc_error(-32600, "Invalid Request", Variant()));
			continue;
		}

		Dictionary request = entry;
		bool should_reply = false;
		Variant response = _handle_jsonrpc_request(request, should_reply);

		if (should_reply) {
			any_reply = true;
			responses.push_back(response);
		}
	}

	r_should_reply = any_reply;
	return responses;
}

Dictionary SceneTreeMCP::_handle_mcp_initialize(const Dictionary &p_request, const Variant &p_id) {
	String requested_protocol = "";
	if (p_request.has("params")) {
		Variant params_var = p_request.get("params", Variant());
		if (params_var.get_type() == Variant::DICTIONARY) {
			Dictionary params = params_var;
			if (params.has("protocolVersion")) {
				requested_protocol = String(params.get("protocolVersion", Variant()));
			}
		}
	}

	if (!requested_protocol.is_empty() && !_is_supported_protocol_version(requested_protocol)) {
		return _make_jsonrpc_error(-32602, "Unsupported protocolVersion: " + requested_protocol, p_id);
	}

	negotiated_protocol_version = requested_protocol.is_empty() ? String("2025-03-26") : requested_protocol;

	Dictionary result;
	result["protocolVersion"] = negotiated_protocol_version;

	Dictionary capabilities;
	Dictionary tools_cap;
	tools_cap["listChanged"] = true;
	capabilities["tools"] = tools_cap;
	Dictionary resources_cap;
	resources_cap["subscribe"] = false;
	resources_cap["listChanged"] = true;
	capabilities["resources"] = resources_cap;
	Dictionary prompts_cap;
	prompts_cap["listChanged"] = true;
	capabilities["prompts"] = prompts_cap;
	result["capabilities"] = capabilities;

	Dictionary serverInfo;
	serverInfo["name"] = server_info_name;
	serverInfo["version"] = server_info_version;
	result["serverInfo"] = serverInfo;

	return _make_jsonrpc_result(result, p_id);
}

Dictionary SceneTreeMCP::_handle_mcp_initialized(const Dictionary &p_request, const Variant &p_id) {
	is_initialized = true;
	// This is a notification, handled in _handle_jsonrpc_request
	return _make_jsonrpc_result(Variant(), p_id);
}

Dictionary SceneTreeMCP::_handle_mcp_tools_list(const Dictionary &p_request, const Variant &p_id) {
	Array tools_array;
	for (const auto &E : tool_registry) {
		Dictionary tool;
		tool["name"] = E.value.name;
		tool["description"] = E.value.description;
		tool["inputSchema"] = E.value.schema;
		tools_array.push_back(tool);
	}

	Dictionary result;
	result["tools"] = tools_array;
	return _make_jsonrpc_result(result, p_id);
}

Dictionary SceneTreeMCP::_handle_mcp_tools_call(const Dictionary &p_request, const Variant &p_id) {
	if (!p_request.has("params")) {
		return _make_jsonrpc_error(-32602, "Invalid params", p_id);
	}

	Variant params_var = p_request.get("params", Variant());
	if (params_var.get_type() != Variant::DICTIONARY) {
		return _make_jsonrpc_error(-32602, "Invalid params", p_id);
	}

	Dictionary params = params_var;
	if (!params.has("name")) {
		return _make_jsonrpc_error(-32602, "Missing tool name", p_id);
	}

	String tool_name = params.get("name", Variant());
	if (!tool_registry.has(tool_name)) {
		return _make_jsonrpc_error(-32602, "Tool not found: " + tool_name, p_id);
	}

	Array args;
	if (params.has("arguments")) {
		Variant args_var = params.get("arguments", Variant());
		if (args_var.get_type() == Variant::ARRAY) {
			args = args_var;
		}
	}
	Variant result = tool_registry[tool_name].callable.callv(args);

	Dictionary tool_result;
	Array content_array;
	Dictionary content_item;
	content_item["type"] = "text";
	content_item["text"] = result.stringify();
	content_array.push_back(content_item);
	tool_result["content"] = content_array;
	tool_result["isError"] = false;

	return _make_jsonrpc_result(tool_result, p_id);
}

Dictionary SceneTreeMCP::_handle_mcp_resources_list(const Dictionary &p_request, const Variant &p_id) {
	Array resources_array;
	for (const auto &E : resource_registry) {
		Dictionary resource;
		resource["uri"] = E.value.uri;
		resource["name"] = E.value.name;
		resources_array.push_back(resource);
	}

	Dictionary result;
	result["resources"] = resources_array;
	return _make_jsonrpc_result(result, p_id);
}

Dictionary SceneTreeMCP::_handle_mcp_resources_read(const Dictionary &p_request, const Variant &p_id) {
	if (!p_request.has("params")) {
		return _make_jsonrpc_error(-32602, "Invalid params", p_id);
	}

	Variant params_var = p_request.get("params", Variant());
	if (params_var.get_type() != Variant::DICTIONARY) {
		return _make_jsonrpc_error(-32602, "Invalid params", p_id);
	}

	Dictionary params = params_var;
	if (!params.has("uri")) {
		return _make_jsonrpc_error(-32602, "Missing resource URI", p_id);
	}

	String uri = params.get("uri", Variant());
	if (!resource_registry.has(uri)) {
		return _make_jsonrpc_error(-32602, "Resource not found: " + uri, p_id);
	}

	Variant result = resource_registry[uri].callable.callv(Array());

	Dictionary resource_result;
	Array contents_array;
	Dictionary content_item;
	content_item["uri"] = uri;
	content_item["mimeType"] = "text/plain";
	content_item["text"] = result.stringify();
	contents_array.push_back(content_item);
	resource_result["contents"] = contents_array;

	return _make_jsonrpc_result(resource_result, p_id);
}

Dictionary SceneTreeMCP::_handle_mcp_prompts_list(const Dictionary &p_request, const Variant &p_id) {
	Array prompts_array;
	for (const auto &E : prompt_registry) {
		Dictionary prompt;
		prompt["name"] = E.value.name;
		prompt["description"] = E.value.description;
		prompt["arguments"] = E.value.arguments;
		prompts_array.push_back(prompt);
	}

	Dictionary result;
	result["prompts"] = prompts_array;
	return _make_jsonrpc_result(result, p_id);
}

Dictionary SceneTreeMCP::_handle_mcp_prompts_get(const Dictionary &p_request, const Variant &p_id) {
	if (!p_request.has("params")) {
		return _make_jsonrpc_error(-32602, "Invalid params", p_id);
	}

	Variant params_var = p_request.get("params", Variant());
	if (params_var.get_type() != Variant::DICTIONARY) {
		return _make_jsonrpc_error(-32602, "Invalid params", p_id);
	}

	Dictionary params = params_var;
	if (!params.has("name")) {
		return _make_jsonrpc_error(-32602, "Missing prompt name", p_id);
	}

	String prompt_name = params.get("name", Variant());
	if (!prompt_registry.has(prompt_name)) {
		return _make_jsonrpc_error(-32602, "Prompt not found: " + prompt_name, p_id);
	}

	const RegisteredPrompt &prompt = prompt_registry[prompt_name];

	Dictionary result;
	result["description"] = prompt.description;

	Array messages;
	Dictionary message;
	message["role"] = "user";
	Dictionary content;
	content["type"] = "text";
	content["text"] = prompt.template_text;
	message["content"] = content;
	messages.push_back(message);
	result["messages"] = messages;

	return _make_jsonrpc_result(result, p_id);
}

void SceneTreeMCP::_collect_project_filesystem_entries(const String &p_dir, PackedStringArray &r_entries) const {
	PackedStringArray directories = DirAccess::get_directories_at(p_dir);
	for (int i = 0; i < directories.size(); i++) {
		const String dir_name = directories[i];
		if (dir_name == ".godot" || dir_name == ".git") {
			continue;
		}
		const String subdir = p_dir.path_join(dir_name);
		_collect_project_filesystem_entries(subdir, r_entries);
	}

	PackedStringArray files = DirAccess::get_files_at(p_dir);
	for (int i = 0; i < files.size(); i++) {
		const String file_path = p_dir.path_join(files[i]);
		if (_is_likely_binary_file(file_path)) {
			continue;
		}
		r_entries.push_back(file_path);
	}
}

bool SceneTreeMCP::_is_likely_binary_file(const String &p_path) const {
	const String ext = p_path.get_extension().to_lower();
	static const char *binary_exts[] = {
		"png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "svgz", "exr", "hdr", "ktx", "dds",
		"ogg", "mp3", "wav", "flac", "m4a", "aac",
		"ttf", "otf", "woff", "woff2",
		"zip", "pck", "exe", "dll", "so", "dylib", "a", "o", "obj", "bin", "wasm", "class", "pyc",
		"res", "import"
	};
	for (const char *binary_ext : binary_exts) {
		if (ext == binary_ext) {
			return true;
		}
	}

	Error err = OK;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err != OK || file.is_null()) {
		return true;
	}

	const uint64_t probe_size = MIN((uint64_t)4096, (uint64_t)file->get_length());
	PackedByteArray sample = file->get_buffer(probe_size);
	for (int i = 0; i < sample.size(); i++) {
		if (sample[i] == 0) {
			return true;
		}
	}

	return false;
}

String SceneTreeMCP::_read_project_filesystem_resource() {
	PackedStringArray entries;
	_collect_project_filesystem_entries("res://", entries);
	entries.sort();

	String output;
	for (int i = 0; i < entries.size(); i++) {
		output += entries[i] + "\n";
	}

	return output;
}

Dictionary SceneTreeMCP::_handle_mcp_ping(const Dictionary &p_request, const Variant &p_id) {
	return _make_jsonrpc_result(Variant(), p_id);
}

bool SceneTreeMCP::_is_supported_protocol_version(const String &p_version) const {
	return p_version == "2025-03-26" || p_version == "2024-11-05";
}

Dictionary SceneTreeMCP::_make_jsonrpc_error(int p_code, const String &p_message, const Variant &p_id) {
	Dictionary response;
	response["jsonrpc"] = "2.0";

	Dictionary error;
	error["code"] = p_code;
	error["message"] = p_message;
	response["error"] = error;

	response["id"] = p_id;
	return response;
}

Dictionary SceneTreeMCP::_make_jsonrpc_result(const Variant &p_result, const Variant &p_id) {
	Dictionary response;
	response["jsonrpc"] = "2.0";
	response["result"] = p_result;
	response["id"] = p_id;
	return response;
}

String SceneTreeMCP::_get_content_type(const Dictionary &p_headers) {
	// Check if client requested JSON Lines format (streaming)
	if (p_headers.has("accept")) {
		String accept = String(p_headers.get("accept", "")).to_lower();
		if (accept.contains("application/jsonl") || accept.contains("application/json-seq")) {
			return "application/jsonl";
		}
	}
	// Check Content-Type header for incoming request
	if (p_headers.has("content-type")) {
		String content_type = String(p_headers.get("content-type", "")).to_lower();
		if (content_type.contains("application/jsonl") || content_type.contains("application/json-seq")) {
			return "application/jsonl";
		}
	}
	// Default to standard JSON
	return "application/json";
}

bool SceneTreeMCP::_is_streaming_request(const Dictionary &p_headers) {
	// Check if client expects streaming responses
	if (p_headers.has("accept")) {
		String accept = String(p_headers.get("accept", "")).to_lower();
		if (accept.contains("application/jsonl") || accept.contains("application/json-seq")) {
			return true;
		}
	}
	return false;
}

String SceneTreeMCP::_format_jsonl_response(const Variant &p_response, const Variant &p_id) {
	// Format a single JSON-RPC response as JSON Lines (one line per response)
	JSON json;
	Dictionary response_dict = _make_jsonrpc_result(p_response, p_id);
	return json.stringify(response_dict) + "\n";
}
