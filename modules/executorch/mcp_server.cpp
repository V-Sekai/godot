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
#include "core/object/class_db.h"

MCPServer::MCPServer() {
	server_running = false;
	port = 8080;
	server_name = "Godot MCP Server";
	
	// Initialize MCP capabilities
	capabilities = Dictionary();
	tools = Array();
	resources = Array();
	
	initialize_mcp();
}

MCPServer::~MCPServer() {
	if (server_running) {
		stop_server();
	}
}

void MCPServer::_bind_methods() {
	// Property bindings
	ClassDB::bind_method(D_METHOD("set_port", "port"), &MCPServer::set_port);
	ClassDB::bind_method(D_METHOD("get_port"), &MCPServer::get_port);
	ClassDB::bind_method(D_METHOD("set_server_name", "name"), &MCPServer::set_server_name);
	ClassDB::bind_method(D_METHOD("get_server_name"), &MCPServer::get_server_name);
	
	// Server control methods
	ClassDB::bind_method(D_METHOD("start_server", "port"), &MCPServer::start_server, DEFVAL(8080));
	ClassDB::bind_method(D_METHOD("stop_server"), &MCPServer::stop_server);
	ClassDB::bind_method(D_METHOD("is_server_running"), &MCPServer::is_server_running);
	
	// MCP protocol methods
	ClassDB::bind_method(D_METHOD("add_tool", "name", "description", "schema"), &MCPServer::add_tool);
	ClassDB::bind_method(D_METHOD("add_resource", "uri", "name", "description"), &MCPServer::add_resource);
	ClassDB::bind_method(D_METHOD("handle_request", "request"), &MCPServer::handle_request);
	
	// Properties
	ADD_PROPERTY(PropertyInfo(Variant::INT, "port"), "set_port", "get_port");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "server_name"), "set_server_name", "get_server_name");
	
	// Signals
	ADD_SIGNAL(MethodInfo("client_connected"));
	ADD_SIGNAL(MethodInfo("client_disconnected"));
	ADD_SIGNAL(MethodInfo("message_received", PropertyInfo(Variant::DICTIONARY, "message")));
	ADD_SIGNAL(MethodInfo("tool_called", PropertyInfo(Variant::STRING, "tool_name"), PropertyInfo(Variant::DICTIONARY, "arguments")));
}

void MCPServer::_ready() {
	print_line("MCP Server node ready");
}

void MCPServer::_exit_tree() {
	if (server_running) {
		stop_server();
	}
}

void MCPServer::start_server(int p_port) {
	if (server_running) {
		print_line("MCP Server already running");
		return;
	}
	
	port = p_port;
	server_running = true;
	
	print_line("MCP Server started on port " + String::num(port));
	// TODO: Implement actual server socket creation and listening
}

void MCPServer::stop_server() {
	if (!server_running) {
		return;
	}
	
	server_running = false;
	print_line("MCP Server stopped");
	// TODO: Implement actual server shutdown
}

bool MCPServer::is_server_running() const {
	return server_running;
}

void MCPServer::initialize_mcp() {
	// Initialize MCP protocol capabilities
	capabilities[String("tools")] = Dictionary();
	capabilities[String("resources")] = Dictionary();
	capabilities[String("prompts")] = Dictionary();
	capabilities[String("logging")] = Dictionary();
	
	print_line("MCP Server initialized with default capabilities");
}

void MCPServer::add_tool(const String &name, const String &description, const Dictionary &schema) {
	Dictionary tool;
	tool[String("name")] = name;
	tool[String("description")] = description;
	tool[String("inputSchema")] = schema;
	
	tools.append(tool);
	print_line("Added MCP tool: " + name);
}

void MCPServer::add_resource(const String &uri, const String &name, const String &description) {
	Dictionary resource;
	resource[String("uri")] = uri;
	resource[String("name")] = name;
	resource[String("description")] = description;
	
	resources.append(resource);
	print_line("Added MCP resource: " + name);
}

Dictionary MCPServer::handle_request(const Dictionary &request) {
	Dictionary response;
	
	if (!request.has("method")) {
		response[String("error")] = "Missing method in request";
		return response;
	}
	
	String method = request["method"];
	
	if (method == "initialize") {
		Dictionary result;
		result[String("capabilities")] = capabilities;
		
		Dictionary server_info;
		server_info[String("name")] = server_name;
		server_info[String("version")] = "1.0.0";
		result[String("serverInfo")] = server_info;
		
		response[String("result")] = result;
	} else if (method == "tools/list") {
		Dictionary result;
		result[String("tools")] = tools;
		response[String("result")] = result;
	} else if (method == "resources/list") {
		Dictionary result;
		result[String("resources")] = resources;
		response[String("result")] = result;
	} else {
		response[String("error")] = "Unknown method: " + method;
	}
	
	return response;
}

void MCPServer::set_port(int p_port) {
	port = p_port;
}

int MCPServer::get_port() const {
	return port;
}

void MCPServer::set_server_name(const String &p_name) {
	server_name = p_name;
}

String MCPServer::get_server_name() const {
	return server_name;
}

void MCPServer::_on_client_connected() {
	emit_signal("client_connected");
}

void MCPServer::_on_client_disconnected() {
	emit_signal("client_disconnected");
}

void MCPServer::_on_message_received(const Dictionary &message) {
	emit_signal("message_received", message);
}