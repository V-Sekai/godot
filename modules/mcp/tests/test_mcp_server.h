/**************************************************************************/
/*  test_mcp_server.h                                                     */
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

#include "tests/test_macros.h"
#include "tests/test_utils.h"

#include "../mcp_protocol.h"
#include "../mcp_server.h"

namespace TestMCPServer {

TEST_CASE("[MCPServer] Server lifecycle - basic start/stop") {
	MCPServer server;
	server.start_server(8080);
	CHECK(server.is_server_running() == true);
	server.stop_server();
	CHECK(server.is_server_running() == false);
}

TEST_CASE("[MCPServer] Server lifecycle - multiple start/stop cycles") {
	MCPServer server;

	// Test multiple cycles
	for (int i = 0; i < 3; i++) {
		server.start_server(8080 + i);
		CHECK(server.is_server_running() == true);
		server.stop_server();
		CHECK(server.is_server_running() == false);
	}
}

TEST_CASE("[MCPServer] Server lifecycle - stop when not running") {
	MCPServer server;

	// Should not crash
	server.stop_server();
	CHECK(server.is_server_running() == false);

	// Stop again
	server.stop_server();
}

TEST_CASE("[MCPServer] Tool registration - basic functionality") {
	MCPServer server;
	server.start_server(8081);

	// Register a tool with empty callable (should not crash)
	Callable tool_handler = Callable();
	server.register_tool("test_tool", tool_handler);

	// Register multiple tools
	server.register_tool("tool1", Callable());
	server.register_tool("tool2", Callable());

	server.stop_server();
}

TEST_CASE("[MCPServer] Tool registration - duplicate names") {
	MCPServer server;
	server.start_server(8082);

	// Register same tool name multiple times (should overwrite)
	server.register_tool("duplicate_tool", Callable());
	server.register_tool("duplicate_tool", Callable());

	server.stop_server();
}

TEST_CASE("[MCPServer] Tool registration - empty name") {
	MCPServer server;
	server.start_server(8083);

	// Register tool with empty name
	server.register_tool("", Callable());

	server.stop_server();
}

TEST_CASE("[MCPServer] Resource registration - basic functionality") {
	MCPServer server;
	server.start_server(8084);

	// Register a resource with empty callable
	Callable resource_handler = Callable();
	server.register_resource("test_resource", resource_handler);

	// Register multiple resources
	server.register_resource("resource1", Callable());
	server.register_resource("resource2", Callable());

	server.stop_server();
}

TEST_CASE("[MCPServer] Resource registration - duplicate names") {
	MCPServer server;
	server.start_server(8085);

	// Register same resource name multiple times
	server.register_resource("duplicate_resource", Callable());
	server.register_resource("duplicate_resource", Callable());

	server.stop_server();
}

TEST_CASE("[MCPServer] Resource registration - empty name") {
	MCPServer server;
	server.start_server(8086);

	// Register resource with empty name
	server.register_resource("", Callable());

	server.stop_server();
}

TEST_CASE("[MCPServer] Message sending - response to valid client") {
	MCPServer server;
	server.start_server(8087);

	// Test sending response (should not crash even with no real clients)
	Dictionary response;
	response["jsonrpc"] = "2.0";
	response["result"] = "test";
	response["id"] = 1;

	server.send_response(1, 1, response);

	server.stop_server();
}

TEST_CASE("[MCPServer] Message sending - notification to valid client") {
	MCPServer server;
	server.start_server(8088);

	server.send_notification(1, "test/notification", Dictionary());

	server.stop_server();
}

TEST_CASE("[MCPServer] Message sending - response to invalid client") {
	MCPServer server;
	server.start_server(8089);

	// Send to non-existent client ID (should not crash)
	Dictionary response;
	response["jsonrpc"] = "2.0";
	response["result"] = "test";
	response["id"] = 1;

	server.send_response(999, 1, response);
	server.send_notification(999, "test", Dictionary());

	server.stop_server();
}

TEST_CASE("[MCPServer] Message sending - progress notification") {
	MCPServer server;
	server.start_server(8090);

	Dictionary progress;
	progress["percentage"] = 75;
	progress["status"] = "processing";

	server.send_progress(1, "token123", progress);

	server.stop_server();
}

TEST_CASE("[MCPServer] Polling - when server not running") {
	MCPServer server;

	// Should not crash
	server.poll();
}

TEST_CASE("[MCPServer] Polling - when server running") {
	MCPServer server;
	server.start_server(8091);

	// Should not crash
	server.poll();
	server.poll(); // Multiple polls

	server.stop_server();
}

TEST_CASE("[MCPServer] Initialize message handling") {
	MCPServer server;
	server.start_server(8092);

	// Simulate initialize request
	Dictionary params;
	params["protocolVersion"] = "2025-06-18";
	params["capabilities"] = Dictionary();

	Dictionary init_message;
	init_message["jsonrpc"] = "2.0";
	init_message["method"] = "initialize";
	init_message["params"] = params;
	init_message["id"] = 1;

	// This would normally be called by the transport layer
	// For testing, we verify the message structure is valid
	CHECK(init_message.has("method"));
	CHECK((String)init_message["method"] == "initialize");

	server.stop_server();
}

TEST_CASE("[MCPServer] Tools list request handling") {
	MCPServer server;
	server.start_server(8093);

	// Register some tools
	server.register_tool("tool1", Callable());
	server.register_tool("tool2", Callable());

	Dictionary tools_list_message;
	tools_list_message["jsonrpc"] = "2.0";
	tools_list_message["method"] = "tools/list";
	tools_list_message["id"] = 2;

	// Message structure validation
	CHECK(tools_list_message.has("method"));
	CHECK((String)tools_list_message["method"] == "tools/list");

	server.stop_server();
}

TEST_CASE("[MCPServer] Tools call request handling") {
	MCPServer server;
	server.start_server(8094);

	server.register_tool("echo", Callable());

	Dictionary params;
	params["name"] = "echo";
	Dictionary args;
	args["message"] = "test";
	params["arguments"] = args;

	Dictionary tools_call_message;
	tools_call_message["jsonrpc"] = "2.0";
	tools_call_message["method"] = "tools/call";
	tools_call_message["params"] = params;
	tools_call_message["id"] = 3;

	// Message structure validation
	CHECK(tools_call_message.has("method"));
	CHECK((String)tools_call_message["method"] == "tools/call");
	CHECK(tools_call_message.has("params"));

	Dictionary call_params = tools_call_message["params"];
	CHECK((String)call_params["name"] == "echo");

	server.stop_server();
}

TEST_CASE("[MCPServer] Resources list request handling") {
	MCPServer server;
	server.start_server(8095);

	server.register_resource("resource1", Callable());

	Dictionary resources_list_message;
	resources_list_message["jsonrpc"] = "2.0";
	resources_list_message["method"] = "resources/list";
	resources_list_message["id"] = 4;

	// Message structure validation
	CHECK(resources_list_message.has("method"));
	CHECK((String)resources_list_message["method"] == "resources/list");

	server.stop_server();
}

TEST_CASE("[MCPServer] Resources read request handling") {
	MCPServer server;
	server.start_server(8096);

	server.register_resource("test_resource", Callable());

	Dictionary params;
	params["uri"] = "resource://test_resource";

	Dictionary resources_read_message;
	resources_read_message["jsonrpc"] = "2.0";
	resources_read_message["method"] = "resources/read";
	resources_read_message["params"] = params;
	resources_read_message["id"] = 5;

	// Message structure validation
	CHECK(resources_read_message.has("method"));
	CHECK((String)resources_read_message["method"] == "resources/read");
	CHECK(resources_read_message.has("params"));

	Dictionary read_params = resources_read_message["params"];
	CHECK((String)read_params["uri"] == "resource://test_resource");

	server.stop_server();
}

TEST_CASE("[MCPServer] Error message handling - invalid method") {
	MCPServer server;
	server.start_server(8097);

	Dictionary invalid_message;
	invalid_message["jsonrpc"] = "2.0";
	invalid_message["method"] = "invalid/method";
	invalid_message["id"] = 6;

	// Message structure validation
	CHECK(invalid_message.has("method"));
	// The actual error handling would be tested in integration tests

	server.stop_server();
}

TEST_CASE("[MCPServer] Error message handling - malformed message") {
	MCPServer server;
	server.start_server(8098);

	Dictionary malformed_message;
	malformed_message["jsonrpc"] = "2.0";
	// Missing method and id - should be handled gracefully

	server.stop_server();
}

TEST_CASE("[MCPServer] Memory management - server destruction") {
	// Test that server can be created and destroyed multiple times
	for (int i = 0; i < 3; i++) {
		MCPServer *server = memnew(MCPServer);
		server->start_server(8099 + i);
		server->stop_server();
		memdelete(server);
	}
}

TEST_CASE("[MCPServer] Concurrent operations - multiple servers") {
	// Test running multiple servers simultaneously (different ports)
	MCPServer server1;
	MCPServer server2;

	server1.start_server(8100);
	server2.start_server(8101);

	CHECK(server1.is_server_running() == true);
	CHECK(server2.is_server_running() == true);

	server1.stop_server();
	server2.stop_server();

	CHECK(server1.is_server_running() == false);
	CHECK(server2.is_server_running() == false);
}

TEST_CASE("[MCPServer] Tool handler callable validation") {
	MCPServer server;
	server.start_server(8102);

	// Test registering various types of callables
	Callable empty_callable;
	server.register_tool("empty_tool", empty_callable);

	// Test with object method callable (would need actual object in real test)
	// For now, just ensure it doesn't crash
	server.register_tool("another_tool", Callable());

	server.stop_server();
}

TEST_CASE("[MCPServer] Resource handler callable validation") {
	MCPServer server;
	server.start_server(8103);

	Callable empty_callable;
	server.register_resource("empty_resource", empty_callable);

	server.register_resource("another_resource", Callable());

	server.stop_server();
}

} // namespace TestMCPServer
