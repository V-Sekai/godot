/**************************************************************************/
/*  mcp_client.cpp                                                        */
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

#include "mcp_client.h"

#include "core/io/json.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"

void MCPClient::_bind_methods() {
	ClassDB::bind_method(D_METHOD("connect_to_server", "url"), &MCPClient::connect_to_server);
	ClassDB::bind_method(D_METHOD("disconnect_from_server"), &MCPClient::disconnect_from_server);
	ClassDB::bind_method(D_METHOD("is_connected"), &MCPClient::is_connected);

	ClassDB::bind_method(D_METHOD("create_instance", "project_path", "scene_path"), &MCPClient::create_instance, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("start_instance"), &MCPClient::start_instance);
	ClassDB::bind_method(D_METHOD("shutdown_instance"), &MCPClient::shutdown_instance);

	ClassDB::bind_method(D_METHOD("call_method", "node_path", "method_name", "args"), &MCPClient::call_method, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("get_property", "node_path", "property_name"), &MCPClient::get_property);
	ClassDB::bind_method(D_METHOD("set_property", "node_path", "property_name", "value"), &MCPClient::set_property);

	ClassDB::bind_method(D_METHOD("iteration"), &MCPClient::iteration);
}

MCPClient::MCPClient() {
	http_request = nullptr;
	connected = false;
	instance_id = 1;
	next_request_id = 1;
}

MCPClient::~MCPClient() {
	disconnect_from_server();
}

void MCPClient::connect_to_server(const String &url) {
	if (connected) {
		disconnect_from_server();
	}

	server_url = url;
	http_request = memnew(HTTPRequest);
	add_child(http_request);
	http_request->connect("request_completed", callable_mp(this, &MCPClient::_http_request_completed));

	connected = true;
	_initialize_mcp();

	print_line("Connected to MCP server: " + server_url);
}

void MCPClient::disconnect_from_server() {
	if (http_request) {
		http_request->disconnect("request_completed", callable_mp(this, &MCPClient::_http_request_completed));
		remove_child(http_request);
		memdelete(http_request);
		http_request = nullptr;
	}

	connected = false;
	server_url = "";
	print_line("Disconnected from MCP server");
}

void MCPClient::_initialize_mcp() {
	if (!connected || !http_request) {
		return;
	}

	Dictionary params;
	params["capabilities"] = Dictionary();
	params["capabilities"]["tools"] = Dictionary();
	params["capabilities"]["resources"] = Dictionary();

	_send_jsonrpc_request("initialize", params);
}

void MCPClient::_send_jsonrpc_request(const String &method, const Dictionary &params) {
	if (!connected || !http_request) {
		return;
	}

	Dictionary request;
	request["jsonrpc"] = "2.0";
	request["id"] = next_request_id++;
	request["method"] = method;
	request["params"] = params;

	JSON json;
	String json_string = json.stringify(request);

	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");

	String url = server_url + "/jsonrpc";
	Error err = http_request->request(url, headers, HTTPClient::METHOD_POST, json_string);

	if (err != OK) {
		print_line("Failed to send JSON-RPC request: " + itos(err));
	}
}

void MCPClient::_http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != OK) {
		print_line("HTTP request failed with result: " + itos(p_result));
		return;
	}

	if (p_response_code != 200) {
		print_line("HTTP request failed with response code: " + itos(p_response_code));
		return;
	}

	String response_string = String::utf8((const char *)p_body.ptr(), p_body.size());

	JSON json;
	Error err = json.parse(response_string);
	if (err != OK) {
		print_line("Failed to parse JSON response: " + response_string);
		return;
	}

	Variant response = json.get_data();
	if (response.get_type() == Variant::DICTIONARY) {
		Dictionary response_dict = response;
		if (response_dict.has("result")) {
			print_line("MCP Response: " + json.stringify(response_dict["result"]));
		} else if (response_dict.has("error")) {
			print_line("MCP Error: " + json.stringify(response_dict["error"]));
		}
	}
}

void MCPClient::create_instance(const String &project_path, const String &scene_path) {
	if (!connected) {
		print_line("Not connected to MCP server");
		return;
	}

	Dictionary args;
	args["project_path"] = project_path;
	if (!scene_path.is_empty()) {
		args["scene_path"] = scene_path;
	}

	Dictionary params;
	params["name"] = "godot_create_instance";
	params["arguments"] = args;

	_send_jsonrpc_request("tools/call", params);
}

void MCPClient::start_instance() {
	if (!connected) {
		print_line("Not connected to MCP server");
		return;
	}

	Dictionary args;
	args["instance_id"] = instance_id;

	Dictionary params;
	params["name"] = "godot_call_method";
	params["arguments"] = args;

	_send_jsonrpc_request("tools/call", params);
}

void MCPClient::shutdown_instance() {
	if (!connected) {
		print_line("Not connected to MCP server");
		return;
	}

	Dictionary args;
	args["instance_id"] = instance_id;

	Dictionary params;
	params["name"] = "godot_shutdown_instance";
	params["arguments"] = args;

	_send_jsonrpc_request("tools/call", params);
}

void MCPClient::call_method(const String &node_path, const String &method_name, const Array &args) {
	if (!connected) {
		print_line("Not connected to MCP server");
		return;
	}

	Dictionary method_args;
	method_args["instance_id"] = instance_id;
	method_args["node_path"] = node_path;
	method_args["method"] = method_name;
	method_args["args"] = args;

	Dictionary params;
	params["name"] = "godot_call_method";
	params["arguments"] = method_args;

	_send_jsonrpc_request("tools/call", params);
}

Variant MCPClient::get_property(const String &node_path, const String &property_name) {
	if (!connected) {
		print_line("Not connected to MCP server");
		return Variant();
	}

	Dictionary args;
	args["instance_id"] = instance_id;
	args["node_path"] = node_path;
	args["property"] = property_name;

	Dictionary params;
	params["name"] = "godot_get_property";
	params["arguments"] = args;

	_send_jsonrpc_request("tools/call", params);

	// For now, return empty variant - in real implementation we'd wait for response
	return Variant();
}

void MCPClient::set_property(const String &node_path, const String &property_name, const Variant &value) {
	if (!connected) {
		print_line("Not connected to MCP server");
		return;
	}

	Dictionary args;
	args["instance_id"] = instance_id;
	args["node_path"] = node_path;
	args["property"] = property_name;
	args["value"] = value;

	Dictionary params;
	params["name"] = "godot_set_property";
	params["arguments"] = args;

	_send_jsonrpc_request("tools/call", params);
}

void MCPClient::iteration() {
	if (!connected) {
		return;
	}

	Dictionary args;
	args["instance_id"] = instance_id;

	Dictionary params;
	params["name"] = "godot_call_method";
	params["arguments"] = args;

	_send_jsonrpc_request("tools/call", params);
}

void MCPClient::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
			// Could auto-connect here if configured
			break;
		case NOTIFICATION_PROCESS:
			// Handle periodic updates if needed
			break;
	}
}
