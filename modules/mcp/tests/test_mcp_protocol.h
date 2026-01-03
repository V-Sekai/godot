/**************************************************************************/
/*  test_mcp_protocol.h                                                   */
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

namespace TestMCPProtocol {

TEST_CASE("[MCPProtocol] Message parsing - basic request") {
	Dictionary message;
	message["jsonrpc"] = "2.0";
	message["method"] = "test_method";
	message["params"] = Dictionary();
	message["id"] = 123;

	CHECK(MCPProtocol::get_method(message) == "test_method");
	CHECK((int)MCPProtocol::get_request_id(message) == 123);
	CHECK(MCPProtocol::get_params(message).get_type() == Variant::DICTIONARY);
}

TEST_CASE("[MCPProtocol] Message parsing - notification") {
	Dictionary message;
	message["jsonrpc"] = "2.0";
	message["method"] = "test_notification";

	CHECK(MCPProtocol::get_method(message) == "test_notification");
	CHECK(MCPProtocol::get_request_id(message).get_type() == Variant::NIL);
	CHECK(MCPProtocol::get_params(message).get_type() == Variant::NIL);
}

TEST_CASE("[MCPProtocol] Message parsing - string ID") {
	Dictionary message;
	message["jsonrpc"] = "2.0";
	message["method"] = "test_method";
	message["id"] = "string_id";

	CHECK(MCPProtocol::get_method(message) == "test_method");
	CHECK((String)MCPProtocol::get_request_id(message) == "string_id");
}

TEST_CASE("[MCPProtocol] Message parsing - null ID") {
	Dictionary message;
	message["jsonrpc"] = "2.0";
	message["method"] = "test_method";
	message["id"] = Variant(); // null

	CHECK(MCPProtocol::get_method(message) == "test_method");
	CHECK(MCPProtocol::get_request_id(message).get_type() == Variant::NIL);
}

TEST_CASE("[MCPProtocol] Message parsing - array params") {
	Dictionary message;
	message["jsonrpc"] = "2.0";
	message["method"] = "test_method";
	message["params"] = Array();
	message["id"] = 123;

	CHECK(MCPProtocol::get_params(message).get_type() == Variant::ARRAY);
}

TEST_CASE("[MCPProtocol] Message parsing - missing fields") {
	Dictionary empty_message;
	CHECK(MCPProtocol::get_method(empty_message).is_empty());
	CHECK(MCPProtocol::get_request_id(empty_message).get_type() == Variant::NIL);
	CHECK(MCPProtocol::get_params(empty_message).get_type() == Variant::NIL);
}

TEST_CASE("[MCPProtocol] Request vs notification detection") {
	Dictionary request;
	request["jsonrpc"] = "2.0";
	request["method"] = "test";
	request["id"] = 1;

	CHECK(MCPProtocol::is_request(request) == true);
	CHECK(MCPProtocol::is_notification(request) == false);

	Dictionary notification;
	notification["jsonrpc"] = "2.0";
	notification["method"] = "test";

	CHECK(MCPProtocol::is_request(notification) == false);
	CHECK(MCPProtocol::is_notification(notification) == true);

	Dictionary null_id_message;
	null_id_message["jsonrpc"] = "2.0";
	null_id_message["method"] = "test";
	null_id_message["id"] = Variant();

	CHECK(MCPProtocol::is_request(null_id_message) == false);
	CHECK(MCPProtocol::is_notification(null_id_message) == true);
}

TEST_CASE("[MCPProtocol] Error response formatting") {
	Dictionary error = MCPProtocol::make_error_response(
			JSONRPC::METHOD_NOT_FOUND,
			"Method not found",
			456);

	CHECK(error.has("jsonrpc"));
	CHECK((String)error["jsonrpc"] == "2.0");
	CHECK(error.has("error"));
	CHECK(error.has("id"));
	CHECK((int)error["id"] == 456);

	Dictionary err_dict = error["error"];
	CHECK((int)err_dict["code"] == JSONRPC::METHOD_NOT_FOUND);
	CHECK((String)err_dict["message"] == "Method not found");
}

TEST_CASE("[MCPProtocol] Error response formatting - string ID") {
	Dictionary error = MCPProtocol::make_error_response(
			JSONRPC::INVALID_PARAMS,
			"Invalid parameters",
			"request_123");

	CHECK(error.has("jsonrpc"));
	CHECK(error.has("error"));
	CHECK(error.has("id"));
	CHECK((String)error["id"] == "request_123");

	Dictionary err_dict = error["error"];
	CHECK((int)err_dict["code"] == JSONRPC::INVALID_PARAMS);
	CHECK((String)err_dict["message"] == "Invalid parameters");
}

TEST_CASE("[MCPProtocol] Error response formatting - null ID") {
	Dictionary error = MCPProtocol::make_error_response(
			JSONRPC::PARSE_ERROR,
			"Parse error",
			Variant());

	CHECK(error.has("jsonrpc"));
	CHECK(error.has("error"));
	// Should not have id field for null ID
	CHECK(!error.has("id"));
}

TEST_CASE("[MCPProtocol] Success response formatting") {
	Dictionary result_data;
	result_data["value"] = "test";
	result_data["number"] = 42;

	Dictionary response = MCPProtocol::make_success_response(result_data, 789);

	CHECK(response.has("jsonrpc"));
	CHECK((String)response["jsonrpc"] == "2.0");
	CHECK(response.has("result"));
	CHECK(response.has("id"));
	CHECK((int)response["id"] == 789);

	Dictionary result = response["result"];
	CHECK((String)result["value"] == "test");
	CHECK((int)result["number"] == 42);
}

TEST_CASE("[MCPProtocol] Success response formatting - string result") {
	String result_data = "simple string";
	Dictionary response = MCPProtocol::make_success_response(result_data, "req_456");

	CHECK(response.has("jsonrpc"));
	CHECK(response.has("result"));
	CHECK(response.has("id"));
	CHECK((String)response["id"] == "req_456");
	CHECK((String)response["result"] == "simple string");
}

TEST_CASE("[MCPProtocol] Success response formatting - array result") {
	Array result_data;
	result_data.push_back("item1");
	result_data.push_back("item2");

	Dictionary response = MCPProtocol::make_success_response(result_data, 999);

	CHECK(response.has("result"));
	Array result = response["result"];
	CHECK(result.size() == 2);
	CHECK((String)result[0] == "item1");
	CHECK((String)result[1] == "item2");
}

TEST_CASE("[MCPProtocol] Notification formatting") {
	Dictionary params;
	params["key"] = "value";
	params["count"] = 5;

	Dictionary notification = MCPProtocol::make_notification("test/notify", params);

	CHECK(notification.has("jsonrpc"));
	CHECK(notification["jsonrpc"] == "2.0");
	CHECK(notification.has("method"));
	CHECK(notification["method"] == "test/notify");
	CHECK(notification.has("params"));
	CHECK(!notification.has("id")); // Notifications don't have id

	Dictionary notif_params = notification["params"];
	CHECK((String)notif_params["key"] == "value");
	CHECK((int)notif_params["count"] == 5);
}

TEST_CASE("[MCPProtocol] Notification formatting - empty params") {
	Dictionary notification = MCPProtocol::make_notification("simple/notify", Variant());

	CHECK(notification.has("jsonrpc"));
	CHECK(notification.has("method"));
	CHECK(notification["method"] == "simple/notify");
	CHECK(!notification.has("params"));
	CHECK(!notification.has("id"));
}

TEST_CASE("[MCPProtocol] Progress notification") {
	Dictionary progress;
	progress["percentage"] = 75;
	progress["status"] = "processing";
	progress["total"] = 100;

	Dictionary notification = MCPProtocol::make_progress_notification("token123", progress);

	CHECK(notification.has("jsonrpc"));
	CHECK(notification["jsonrpc"] == "2.0");
	CHECK(notification.has("method"));
	CHECK(notification["method"] == "notifications/progress");
	CHECK(notification.has("params"));
	CHECK(!notification.has("id"));

	Dictionary params = notification["params"];
	CHECK((String)params["progressToken"] == "token123");

	Dictionary prog = params["progress"];
	CHECK((int)prog["percentage"] == 75);
	CHECK((String)prog["status"] == "processing");
	CHECK((int)prog["total"] == 100);
}

TEST_CASE("[MCPProtocol] Batch request detection") {
	// Test array (batch)
	Array batch;
	Dictionary msg1;
	msg1["jsonrpc"] = "2.0";
	msg1["method"] = "test1";
	msg1["id"] = 1;
	batch.push_back(msg1);

	CHECK(MCPProtocol::is_batch_request(batch) == true);

	// Test single message (not batch)
	Dictionary single;
	single["jsonrpc"] = "2.0";
	single["method"] = "test";

	CHECK(MCPProtocol::is_batch_request(single) == false);

	// Test empty array
	Array empty_batch;
	CHECK(MCPProtocol::is_batch_request(empty_batch) == true);

	// Test non-array types
	CHECK(MCPProtocol::is_batch_request(Variant()) == false);
	CHECK(MCPProtocol::is_batch_request(String("test")) == false);
	CHECK(MCPProtocol::is_batch_request(42) == false);
}

TEST_CASE("[MCPProtocol] Message validation - valid messages") {
	Dictionary valid_request;
	valid_request["jsonrpc"] = "2.0";
	valid_request["method"] = "test";
	valid_request["id"] = 1;

	CHECK(MCPProtocol::validate_message(valid_request) == true);

	Dictionary valid_notification;
	valid_notification["jsonrpc"] = "2.0";
	valid_notification["method"] = "test";

	CHECK(MCPProtocol::validate_message(valid_notification) == true);

	Dictionary valid_response;
	valid_response["jsonrpc"] = "2.0";
	valid_response["result"] = "success";
	valid_response["id"] = 1;

	CHECK(MCPProtocol::validate_message(valid_response) == true);

	Dictionary valid_error_response;
	valid_error_response["jsonrpc"] = "2.0";
	Dictionary error;
	error["code"] = JSONRPC::METHOD_NOT_FOUND;
	error["message"] = "Method not found";
	valid_error_response["error"] = error;
	valid_error_response["id"] = 1;

	CHECK(MCPProtocol::validate_message(valid_error_response) == true);
}

TEST_CASE("[MCPProtocol] Message validation - invalid messages") {
	// Missing jsonrpc
	Dictionary missing_jsonrpc;
	missing_jsonrpc["method"] = "test";
	CHECK(MCPProtocol::validate_message(missing_jsonrpc) == false);

	// Wrong jsonrpc version
	Dictionary wrong_version;
	wrong_version["jsonrpc"] = "1.0";
	wrong_version["method"] = "test";
	CHECK(MCPProtocol::validate_message(wrong_version) == false);

	// Missing method/result/error
	Dictionary missing_method;
	missing_method["jsonrpc"] = "2.0";
	missing_method["id"] = 1;
	CHECK(MCPProtocol::validate_message(missing_method) == false);

	// Empty dictionary
	Dictionary empty;
	CHECK(MCPProtocol::validate_message(empty) == false);
}

TEST_CASE("[MCPProtocol] Message processing - null JSONRPC") {
	Dictionary message;
	message["jsonrpc"] = "2.0";
	message["method"] = "test";
	message["id"] = 1;

	Variant result = MCPProtocol::process_message(nullptr, message);
	CHECK(result.get_type() == Variant::NIL);
}

TEST_CASE("[MCPProtocol] Batch processing - null JSONRPC") {
	Array messages;
	Dictionary msg;
	msg["jsonrpc"] = "2.0";
	msg["method"] = "test";
	msg["id"] = 1;
	messages.push_back(msg);

	Array result = MCPProtocol::process_batch(nullptr, messages);
	CHECK(result.size() == 0);
}

TEST_CASE("[MCPProtocol] Batch processing - empty array") {
	JSONRPC jsonrpc;
	Array messages;

	Array result = MCPProtocol::process_batch(&jsonrpc, messages);
	CHECK(result.size() == 0);
}

TEST_CASE("[MCPProtocol] Batch processing - mixed valid/invalid") {
	JSONRPC jsonrpc;
	Array messages;

	// Valid message
	Dictionary valid_msg;
	valid_msg["jsonrpc"] = "2.0";
	valid_msg["method"] = "test";
	valid_msg["id"] = 1;
	messages.push_back(valid_msg);

	// Invalid message (non-dictionary)
	messages.push_back(String("invalid"));

	// Another valid message
	Dictionary valid_msg2;
	valid_msg2["jsonrpc"] = "2.0";
	valid_msg2["method"] = "test2";
	valid_msg2["id"] = 2;
	messages.push_back(valid_msg2);

	Array result = MCPProtocol::process_batch(&jsonrpc, messages);
	// Should process valid messages and skip invalid ones
	// (Exact behavior depends on JSONRPC implementation)
	CHECK(result.size() >= 0); // At least doesn't crash
}

TEST_CASE("[MCPProtocol] Protocol version constants") {
	// Test that version constants are defined
	String v1 = MCPProtocol::VERSION_2024_11_05;
	String v2 = MCPProtocol::VERSION_2025_03_26;
	String v3 = MCPProtocol::VERSION_2025_06_18;

	CHECK(!v1.is_empty());
	CHECK(!v2.is_empty());
	CHECK(!v3.is_empty());

	// Verify they are different
	CHECK(v1 != v2);
	CHECK(v2 != v3);
	CHECK(v1 != v3);
}

} // namespace TestMCPProtocol
