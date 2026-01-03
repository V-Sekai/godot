/**************************************************************************/
/*  mcp_protocol.h                                                        */
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

#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "modules/jsonrpc/jsonrpc.h"

class MCPProtocol {
public:
	// MCP protocol versions supported (compatible with ex-mcp)
	static constexpr const char *VERSION_2024_11_05 = "2024-11-05";
	static constexpr const char *VERSION_2025_03_26 = "2025-03-26";
	static constexpr const char *VERSION_2025_06_18 = "2025-06-18";

	// Check if message is request/response (has id) or notification (no id)
	static bool is_request(const Dictionary &p_message);
	static bool is_notification(const Dictionary &p_message);

	// Extract request ID from message
	static Variant get_request_id(const Dictionary &p_message);

	// Extract method name from message
	static String get_method(const Dictionary &p_message);

	// Extract params from message
	static Variant get_params(const Dictionary &p_message);

	// Create MCP error response
	static Dictionary make_error_response(int p_code, const String &p_message, const Variant &p_id = Variant());

	// Create MCP success response
	static Dictionary make_success_response(const Variant &p_result, const Variant &p_id);

	// Create MCP notification
	static Dictionary make_notification(const String &p_method, const Variant &p_params);

	// Create progress notification
	static Dictionary make_progress_notification(const String &p_progress_token, const Dictionary &p_progress);

	// Check if message is a batch request (Array of messages)
	static bool is_batch_request(const Variant &p_message);

	// Validate JSON-RPC message format
	static bool validate_message(const Dictionary &p_message);

	// Process message using JSONRPC
	static Variant process_message(JSONRPC *p_jsonrpc, const Dictionary &p_message);

	// Process batch request (Array of messages)
	static Array process_batch(JSONRPC *p_jsonrpc, const Array &p_messages);
};
