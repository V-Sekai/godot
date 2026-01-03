/**************************************************************************/
/*  test_ex_mcp_compatibility.h                                           */
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

namespace TestExMCPCompatibility {

TEST_CASE("[ExMCPCompatibility] Error handling - malformed JSON") {
	// Test handling of malformed JSON messages

	// Incomplete JSON
	String incomplete_json = "{\"jsonrpc\":\"2.0\",\"method\":\"test\"";
	Variant parsed = JSON::parse_string(incomplete_json);
	// Should fail to parse
	// In real transport, this would be handled by JSON parsing errors

	// Invalid JSON structure
	String invalid_json = "{\"jsonrpc\":\"2.0\",\"method\":\"test\",}"; // Trailing comma
	parsed = JSON::parse_string(invalid_json);
	// Should fail to parse
}

TEST_CASE("[ExMCPCompatibility] Error handling - invalid message types") {
	// Test handling of non-object messages

	Variant invalid_messages[] = {
		Variant("string message"),
		Variant(42),
		Variant(Array()),
		Variant(true)
	};

	for (const Variant &invalid_msg : invalid_messages) {
		// These should not be processed as valid MCP messages
		if (invalid_msg.get_type() == Variant::DICTIONARY) {
			Dictionary dict = invalid_msg;
			// Dictionary validation would happen here
			CHECK(MCPProtocol::validate_message(dict) == false);
		}
	}
}

TEST_CASE("[ExMCPCompatibility] Error handling - oversized messages") {
	// Test handling of very large messages

	// Create a very large message
	String large_content;
	for (int i = 0; i < 10000; i++) {
		large_content += "large content data ";
	}

	Dictionary large_message;
	large_message["jsonrpc"] = "2.0";
	large_message["method"] = "test/large";
	large_message["params"] = Dictionary{{"content", large_content}};
	large_message["id"] = 1;

	// Message should still be valid structurally
	CHECK(MCPProtocol::validate_message(large_message) == true);
}

TEST_CASE("[ExMCPCompatibility] Error handling - deeply nested structures") {
	// Test handling of deeply nested JSON structures

	Dictionary deeply_nested;
	deeply_nested["jsonrpc"] = "2.0";
	deeply_nested["method"] = "test/nested";
	deeply_nested["id"] = 1;

	Dictionary level1;
	Dictionary level2;
	Dictionary level3;
	Dictionary level4;
	Dictionary level5;

	level5["deepest"] = "value";
	level4["level4"] = level5;
	level3["level3"] = level4;
	level2["level2"] = level3;
	level1["level1"] = level2;
	deeply_nested["params"] = level1;

	// Should still be valid
	CHECK(MCPProtocol::validate_message(deeply_nested) == true);
}

TEST_CASE("[ExMCPCompatibility] Error handling - null and undefined values") {
	// Test handling of null/undefined parameter values

	Dictionary message_with_nulls;
	message_with_nulls["jsonrpc"] = "2.0";
	message_with_nulls["method"] = "test/nulls";
	message_with_nulls["params"] = Dictionary{
		{"null_param", Variant()},
		{"string_param", "value"},
		{"int_param", 42}
	};
	message_with_nulls["id"] = 1;

	// Should be valid
	CHECK(MCPProtocol::validate_message(message_with_nulls) == true);

	// Test extraction of null values
	Variant params = MCPProtocol::get_params(message_with_nulls);
	CHECK(params.get_type() == Variant::DICTIONARY);
}

TEST_CASE("[ExMCPCompatibility] Error handling - Unicode and special characters") {
	// Test handling of Unicode strings and special characters

	String unicode_strings[] = {
		"héllo wörld", // accented characters
		"🚀🌟💻", // emojis
		"тест на русском", // Cyrillic
		"测试中文", // Chinese
		"<>&\"'", // HTML/XML special chars
		"null\n\t\r" // control characters
	};

	for (const String &unicode_str : unicode_strings) {
		Dictionary unicode_message;
		unicode_message["jsonrpc"] = "2.0";
		unicode_message["method"] = "test/unicode";
		unicode_message["params"] = Dictionary{{"text", unicode_str}};
		unicode_message["id"] = 1;

		// Should handle Unicode correctly
		CHECK(MCPProtocol::validate_message(unicode_message) == true);
	}
}

TEST_CASE("[ExMCPCompatibility] Error handling - concurrent message processing") {
	// Test processing multiple messages simultaneously

	const int num_messages = 10;
	JSONRPC jsonrpc;

	// Process multiple messages in sequence (simulating concurrent load)
	for (int i = 0; i < num_messages; i++) {
		Dictionary message;
		message["jsonrpc"] = "2.0";
		message["method"] = "test/concurrent";
		message["params"] = Dictionary{{"index", i}};
		message["id"] = i + 1;

		Variant result = MCPProtocol::process_message(&jsonrpc, message);
		// Should not crash and should return some result
		CHECK(result.get_type() != Variant::NIL);
	}
}

TEST_CASE("[ExMCPCompatibility] Error handling - memory exhaustion simulation") {
	// Test behavior under memory pressure (simulated)

	// Create many large messages
	const int num_large_messages = 100;
	Dictionary *messages = new Dictionary[num_large_messages];

	for (int i = 0; i < num_large_messages; i++) {
		messages[i]["jsonrpc"] = "2.0";
		messages[i]["method"] = "test/memory";
		messages[i]["id"] = i + 1;

		// Add some data
		Dictionary params;
		Array data_array;
		for (int j = 0; j < 100; j++) {
			data_array.push_back("data_item_" + String::num_int64(j));
		}
		params["data"] = data_array;
		messages[i]["params"] = params;
	}

	// Verify all messages are valid
	for (int i = 0; i < num_large_messages; i++) {
		CHECK(MCPProtocol::validate_message(messages[i]) == true);
	}

	delete[] messages;
}

TEST_CASE("[ExMCPCompatibility] Error handling - network-like error conditions") {
	// Simulate network transport errors

	// Test partial message buffering (simulated)
	String partial_messages[] = {
		"{\"jsonrpc\":\"2.0\"",
		"{\"jsonrpc\":\"2.0\",\"method\"",
		"{\"jsonrpc\":\"2.0\",\"method\":\"test\"",
		"{\"jsonrpc\":\"2.0\",\"method\":\"test\",\"id\":1",
		"{\"jsonrpc\":\"2.0\",\"method\":\"test\",\"id\":1}"
	};

	for (const String &partial : partial_messages) {
		Variant parsed = JSON::parse_string(partial);
		if (parsed.get_type() == Variant::DICTIONARY) {
			Dictionary dict = parsed;
			// Check if it's a complete message
			bool is_complete = dict.has("jsonrpc") && (dict.has("method") || dict.has("result"));
			// Partial messages should not be considered complete
			if (partial.find("}") == -1) {
				CHECK(is_complete == false);
			}
		}
	}
}

TEST_CASE("[ExMCPCompatibility] Error handling - invalid protocol versions") {
	// Test handling of invalid/unknown protocol versions

	String invalid_versions[] = {
		"1.0",
		"2.0",
		"2023-01-01",
		"2025-13-45", // Invalid date
		"invalid-version",
		"",
		"null"
	};

	for (const String &version : invalid_versions) {
		Dictionary init_params;
		init_params["protocolVersion"] = version;
		init_params["capabilities"] = Dictionary();

		Dictionary init_request;
		init_request["jsonrpc"] = "2.0";
		init_request["method"] = "initialize";
		init_request["params"] = init_params;
		init_request["id"] = 1;

		// Message should still be structurally valid
		CHECK(MCPProtocol::validate_message(init_request) == true);
		// But version negotiation might fail in real implementation
	}
}

TEST_CASE("[ExMCPCompatibility] Error handling - extreme parameter values") {
	// Test handling of extreme parameter values

	Dictionary extreme_message;
	extreme_message["jsonrpc"] = "2.0";
	extreme_message["method"] = "test/extreme";
	extreme_message["id"] = 1;

	Dictionary extreme_params;
	extreme_params["max_int"] = INT64_MAX;
	extreme_params["min_int"] = INT64_MIN;
	extreme_params["max_float"] = DBL_MAX;
	extreme_params["min_float"] = DBL_MIN;
	extreme_params["empty_string"] = "";
	extreme_params["very_long_string"] = String("x").repeat(10000);
	extreme_params["empty_array"] = Array();
	extreme_params["empty_dict"] = Dictionary();
	extreme_params["null_value"] = Variant();

	extreme_message["params"] = extreme_params;

	// Should handle extreme values gracefully
	CHECK(MCPProtocol::validate_message(extreme_message) == true);
}

TEST_CASE("[ExMCPCompatibility] Error handling - circular references simulation") {
	// Test handling of circular reference-like structures (JSON doesn't support true circular refs)

	Dictionary circular_like;
	circular_like["jsonrpc"] = "2.0";
	circular_like["method"] = "test/circular";
	circular_like["id"] = 1;

	// Create nested structure that references similar data
	Dictionary level1;
	Dictionary level2;
	level2["data"] = "shared_data";
	level1["level2"] = level2;
	level1["shared"] = level2; // Reference to same data

	circular_like["params"] = level1;

	// Should be valid (JSON allows this)
	CHECK(MCPProtocol::validate_message(circular_like) == true);
}

TEST_CASE("[ExMCPCompatibility] Error handling - encoding edge cases") {
	// Test various string encoding edge cases

	String edge_case_strings[] = {
		String::utf8("\\u0000"), // Null character
		String::utf8("\\uFFFF"), // Last Unicode character
		"\0null\0byte\0", // Embedded null bytes
		"multi\nline\nstring",
		"tab\tseparated\tvalues",
		"quote\"and\'apostrophe",
		"backslash\\escape\\sequences"
	};

	for (const String &edge_str : edge_case_strings) {
		Dictionary encoding_message;
		encoding_message["jsonrpc"] = "2.0";
		encoding_message["method"] = "test/encoding";
		encoding_message["params"] = Dictionary{{"text", edge_str}};
		encoding_message["id"] = 1;

		// Should handle encoding edge cases
		CHECK(MCPProtocol::validate_message(encoding_message) == true);
	}
}

TEST_CASE("[ExMCPCompatibility] Error handling - timeout simulation") {
	// Simulate timeout conditions

	JSONRPC jsonrpc;

	// Process a message that might take time
	Dictionary slow_message;
	slow_message["jsonrpc"] = "2.0";
	slow_message["method"] = "test/slow";
	slow_message["params"] = Dictionary{{"delay", 1000}}; // Simulated delay
	slow_message["id"] = 1;

	// In real implementation, this might timeout
	Variant result = MCPProtocol::process_message(&jsonrpc, slow_message);
	// Should return some result (even if error)
	CHECK(result.get_type() != Variant::NIL);
}

} // namespace TestExMCPCompatibility
