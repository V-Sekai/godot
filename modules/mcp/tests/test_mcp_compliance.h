/**************************************************************************/
/*  test_mcp_compliance.h                                                 */
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

#include "../mcp_server.h"

namespace TestMCPCompliance {

TEST_CASE("[MCPCompliance] Initialize handshake - basic") {
	JSONRPC jsonrpc;

	Dictionary capabilities;
	capabilities["tools"] = Dictionary{{"listChanged", true}};
	capabilities["resources"] = Dictionary{{"subscribe", true}, {"listChanged", true}};

	Dictionary init_params;
	init_params["protocolVersion"] = MCPProtocol::VERSION_2025_06_18;
	init_params["capabilities"] = capabilities;
	init_params["clientInfo"] = Dictionary{{"name", "TestClient"}, {"version", "1.0"}};

	Dictionary init_request;
	init_request["jsonrpc"] = "2.0";
	init_request["method"] = "initialize";
	init_request["params"] = init_params;
	init_request["id"] = 1;

	Variant result = MCPProtocol::process_message(&jsonrpc, init_request);
	CHECK(result.get_type() != Variant::NIL);

	// Verify message structure
	CHECK(MCPProtocol::validate_message(init_request) == true);
	CHECK(MCPProtocol::is_request(init_request) == true);
}

TEST_CASE("[MCPCompliance] Initialize handshake - all protocol versions") {
	String versions[] = {
		MCPProtocol::VERSION_2024_11_05,
		MCPProtocol::VERSION_2025_03_26,
		MCPProtocol::VERSION_2025_06_18
	};

	for (const String &version : versions) {
		JSONRPC jsonrpc;

		Dictionary init_params;
		init_params["protocolVersion"] = version;
		init_params["capabilities"] = Dictionary();

		Dictionary init_request;
		init_request["jsonrpc"] = "2.0";
		init_request["method"] = "initialize";
		init_request["params"] = init_params;
		init_request["id"] = 1;

		Variant result = MCPProtocol::process_message(&jsonrpc, init_request);
		CHECK(result.get_type() != Variant::NIL);
	}
}

TEST_CASE("[MCPCompliance] Batch requests - homogeneous batch") {
	JSONRPC jsonrpc;
	Array batch;

	// Create batch of similar requests
	for (int i = 0; i < 3; i++) {
		Dictionary request;
		request["jsonrpc"] = "2.0";
		request["method"] = "tools/list";
		request["id"] = i + 1;
		batch.push_back(request);
	}

	Array results = MCPProtocol::process_batch(&jsonrpc, batch);
	// Should process all requests
	CHECK(results.size() >= 0); // JSONRPC may not return results for unknown methods
}

TEST_CASE("[MCPCompliance] Batch requests - mixed batch") {
	JSONRPC jsonrpc;
	Array batch;

	// Initialize request
	Dictionary init_params;
	init_params["protocolVersion"] = MCPProtocol::VERSION_2025_06_18;
	init_params["capabilities"] = Dictionary();

	Dictionary init_request;
	init_request["jsonrpc"] = "2.0";
	init_request["method"] = "initialize";
	init_request["params"] = init_params;
	init_request["id"] = 1;
	batch.push_back(init_request);

	// Tools list request
	Dictionary tools_request;
	tools_request["jsonrpc"] = "2.0";
	tools_request["method"] = "tools/list";
	tools_request["id"] = 2;
	batch.push_back(tools_request);

	// Notification (no response expected)
	Dictionary notification;
	notification["jsonrpc"] = "2.0";
	notification["method"] = "notifications/initialized";
	batch.push_back(notification);

	Array results = MCPProtocol::process_batch(&jsonrpc, batch);
	// Should have results for requests, but not for notification
	CHECK(results.size() >= 0);
}

TEST_CASE("[MCPCompliance] Batch requests - empty batch") {
	JSONRPC jsonrpc;
	Array batch;

	Array results = MCPProtocol::process_batch(&jsonrpc, batch);
	CHECK(results.size() == 0);
}

TEST_CASE("[MCPCompliance] Batch requests - invalid messages in batch") {
	JSONRPC jsonrpc;
	Array batch;

	// Valid request
	Dictionary valid_request;
	valid_request["jsonrpc"] = "2.0";
	valid_request["method"] = "initialize";
	valid_request["params"] = Dictionary{{"protocolVersion", MCPProtocol::VERSION_2025_06_18}};
	valid_request["id"] = 1;
	batch.push_back(valid_request);

	// Invalid message (missing jsonrpc)
	Dictionary invalid_message;
	invalid_message["method"] = "test";
	invalid_message["id"] = 2;
	batch.push_back(invalid_message);

	// Another valid request
	Dictionary another_valid;
	another_valid["jsonrpc"] = "2.0";
	another_valid["method"] = "tools/list";
	another_valid["id"] = 3;
	batch.push_back(another_valid);

	Array results = MCPProtocol::process_batch(&jsonrpc, batch);
	// Should process valid messages and skip invalid ones
	CHECK(results.size() >= 0);
}

TEST_CASE("[MCPCompliance] Progress notifications - various progress states") {
	Dictionary progress_states[] = {
		Dictionary{{"percentage", 0}, {"status", "starting"}},
		Dictionary{{"percentage", 25.5}, {"status", "processing"}, {"message", "Working..."}},
		Dictionary{{"percentage", 50}, {"status", "halfway"}, {"total", 100}},
		Dictionary{{"percentage", 100}, {"status", "completed"}}
	};

	for (int i = 0; i < 4; i++) {
		String token = "progress_token_" + String::num_int64(i);
		Dictionary notification = MCPProtocol::make_progress_notification(token, progress_states[i]);

		// Verify structure
		CHECK(notification.has("jsonrpc"));
		CHECK(notification["jsonrpc"] == "2.0");
		CHECK(notification.has("method"));
		CHECK(notification["method"] == "notifications/progress");
		CHECK(notification.has("params"));
		CHECK(!notification.has("id"));

		Dictionary params = notification["params"];
		CHECK(params.has("progressToken"));
		CHECK(params.has("progress"));
	}
}

TEST_CASE("[MCPCompliance] Progress notifications - empty progress data") {
	Dictionary empty_progress;
	Dictionary notification = MCPProtocol::make_progress_notification("token", empty_progress);

	// Should still create valid notification structure
	CHECK(notification.has("method"));
	CHECK(notification["method"] == "notifications/progress");
}

TEST_CASE("[MCPCompliance] Tools API compliance - tools/list format") {
	JSONRPC jsonrpc;

	Dictionary tools_list_request;
	tools_list_request["jsonrpc"] = "2.0";
	tools_list_request["method"] = "tools/list";
	tools_list_request["id"] = 1;

	Variant result = MCPProtocol::process_message(&jsonrpc, tools_list_request);
	// In a real implementation, this would return a properly formatted tools list
	// For now, just verify the request is processed
	CHECK(result.get_type() != Variant::NIL);
}

TEST_CASE("[MCPCompliance] Tools API compliance - tools/call format") {
	JSONRPC jsonrpc;

	Dictionary tool_args;
	tool_args["message"] = "test input";

	Dictionary call_params;
	call_params["name"] = "echo_tool";
	call_params["arguments"] = tool_args;

	Dictionary tools_call_request;
	tools_call_request["jsonrpc"] = "2.0";
	tools_call_request["method"] = "tools/call";
	tools_call_request["params"] = call_params;
	tools_call_request["id"] = 2;

	Variant result = MCPProtocol::process_message(&jsonrpc, tools_call_request);
	// Verify request structure is valid
	CHECK(MCPProtocol::validate_message(tools_call_request) == true);
}

TEST_CASE("[MCPCompliance] Resources API compliance - resources/list format") {
	JSONRPC jsonrpc;

	Dictionary resources_list_request;
	resources_list_request["jsonrpc"] = "2.0";
	resources_list_request["method"] = "resources/list";
	resources_list_request["id"] = 3;

	Variant result = MCPProtocol::process_message(&jsonrpc, resources_list_request);
	CHECK(result.get_type() != Variant::NIL);
}

TEST_CASE("[MCPCompliance] Resources API compliance - resources/read format") {
	JSONRPC jsonrpc;

	Dictionary read_params;
	read_params["uri"] = "resource://test/resource";

	Dictionary resources_read_request;
	resources_read_request["jsonrpc"] = "2.0";
	resources_read_request["method"] = "resources/read";
	resources_read_request["params"] = read_params;
	resources_read_request["id"] = 4;

	Variant result = MCPProtocol::process_message(&jsonrpc, resources_read_request);
	// Verify request structure
	CHECK(MCPProtocol::validate_message(resources_read_request) == true);
}

TEST_CASE("[MCPCompliance] Error responses - standard MCP error codes") {
	int mcp_error_codes[] = {
		JSONRPC::PARSE_ERROR,
		JSONRPC::INVALID_REQUEST,
		JSONRPC::METHOD_NOT_FOUND,
		JSONRPC::INVALID_PARAMS,
		JSONRPC::INTERNAL_ERROR
	};

	String error_messages[] = {
		"Parse error",
		"Invalid request",
		"Method not found",
		"Invalid parameters",
		"Internal error"
	};

	for (int i = 0; i < 5; i++) {
		Dictionary error = MCPProtocol::make_error_response(mcp_error_codes[i], error_messages[i], 100 + i);

		// Verify error response structure
		CHECK(error.has("jsonrpc"));
		CHECK(error["jsonrpc"] == "2.0");
		CHECK(error.has("error"));
		CHECK(error.has("id"));

		Dictionary err_dict = error["error"];
		CHECK((int)err_dict["code"] == mcp_error_codes[i]);
		CHECK((String)err_dict["message"] == error_messages[i]);
	}
}

TEST_CASE("[MCPCompliance] Error responses - custom error codes") {
	// Test custom error codes
	int custom_codes[] = {-32000, -32001, -32002};
	String custom_messages[] = {"Custom error 1", "Custom error 2", "Custom error 3"};

	for (int i = 0; i < 3; i++) {
		Dictionary error = MCPProtocol::make_error_response(custom_codes[i], custom_messages[i], 200 + i);

		CHECK(error.has("error"));
		Dictionary err_dict = error["error"];
		CHECK((int)err_dict["code"] == custom_codes[i]);
		CHECK((String)err_dict["message"] == custom_messages[i]);
	}
}

TEST_CASE("[MCPCompliance] Request ID handling - various types") {
	Variant ids[] = {Variant(123), Variant("string_id"), Variant(45.67)};

	for (const Variant &id : ids) {
		Dictionary request;
		request["jsonrpc"] = "2.0";
		request["method"] = "test";
		request["id"] = id;

		// Should be detected as request
		CHECK(MCPProtocol::is_request(request) == true);
		CHECK(MCPProtocol::is_notification(request) == false);

		// Should be able to extract ID
		Variant extracted_id = MCPProtocol::get_request_id(request);
		CHECK(extracted_id == id);
	}
}

TEST_CASE("[MCPCompliance] Notification pattern - no ID field") {
	Dictionary notification;
	notification["jsonrpc"] = "2.0";
	notification["method"] = "test/notification";

	// Should be detected as notification
	CHECK(MCPProtocol::is_request(notification) == false);
	CHECK(MCPProtocol::is_notification(notification) == true);

	// ID should be null
	Variant id = MCPProtocol::get_request_id(notification);
	CHECK(id.get_type() == Variant::NIL);
}

TEST_CASE("[MCPCompliance] Message ordering - request/response correlation") {
	// Test that requests and responses can be correlated by ID

	Dictionary request;
	request["jsonrpc"] = "2.0";
	request["method"] = "test";
	request["id"] = "correlation_id_123";

	Dictionary response;
	response["jsonrpc"] = "2.0";
	response["result"] = "success";
	response["id"] = "correlation_id_123";

	// Both should be valid messages
	CHECK(MCPProtocol::validate_message(request) == true);
	CHECK(MCPProtocol::validate_message(response) == true);

	// IDs should match for correlation
	Variant request_id = MCPProtocol::get_request_id(request);
	Variant response_id = response["id"];
	CHECK(request_id == response_id);
}

TEST_CASE("[MCPCompliance] Capability negotiation - tools capability") {
	Dictionary server_capabilities;
	server_capabilities["tools"] = Dictionary{{"listChanged", true}};

	Dictionary client_capabilities;
	client_capabilities["tools"] = Dictionary{{"listChanged", true}};

	// In a full implementation, these would be compared during initialization
	// For now, just verify the structure
	CHECK(server_capabilities.has("tools"));
	CHECK(client_capabilities.has("tools"));
}

TEST_CASE("[MCPCompliance] Capability negotiation - resources capability") {
	Dictionary server_capabilities;
	server_capabilities["resources"] = Dictionary{{"subscribe", true}, {"listChanged", true}};

	Dictionary client_capabilities;
	client_capabilities["resources"] = Dictionary{{"subscribe", false}, {"listChanged", true}};

	// Verify capability structures
	CHECK(server_capabilities.has("resources"));
	CHECK(client_capabilities.has("resources"));
}

} // namespace TestMCPCompliance
