/**************************************************************************/
/*  mcp_protocol.cpp                                                      */
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

#include "mcp_protocol.h"

#include "modules/jsonrpc/jsonrpc.h"

bool MCPProtocol::is_request(const Dictionary &p_message) {
	return p_message.has("id") && p_message["id"].get_type() != Variant::NIL;
}

bool MCPProtocol::is_notification(const Dictionary &p_message) {
	return !p_message.has("id") || p_message["id"].get_type() == Variant::NIL;
}

Variant MCPProtocol::get_request_id(const Dictionary &p_message) {
	if (p_message.has("id")) {
		return p_message["id"];
	}
	return Variant();
}

String MCPProtocol::get_method(const Dictionary &p_message) {
	if (p_message.has("method")) {
		return p_message["method"];
	}
	return String();
}

Variant MCPProtocol::get_params(const Dictionary &p_message) {
	if (p_message.has("params")) {
		return p_message["params"];
	}
	return Variant();
}

Dictionary MCPProtocol::make_error_response(int p_code, const String &p_message, const Variant &p_id) {
	JSONRPC *jsonrpc = memnew(JSONRPC);
	// Ensure integer IDs are stored as integers, not floats, for JSON-RPC compliance
	Variant id = p_id;
	if (id.get_type() == Variant::FLOAT) {
		double float_val = id;
		int64_t int_val = (int64_t)float_val;
		// Only convert if it's actually an integer (no fractional part)
		if ((double)int_val == float_val) {
			id = int_val;
		}
	}
	Dictionary result = jsonrpc->make_response_error(p_code, p_message, id);
	memdelete(jsonrpc);
	return result;
}

Dictionary MCPProtocol::make_success_response(const Variant &p_result, const Variant &p_id) {
	JSONRPC *jsonrpc = memnew(JSONRPC);
	// Ensure integer IDs are stored as integers, not floats, for JSON-RPC compliance
	Variant id = p_id;
	if (id.get_type() == Variant::FLOAT) {
		double float_val = id;
		int64_t int_val = (int64_t)float_val;
		// Only convert if it's actually an integer (no fractional part)
		if ((double)int_val == float_val) {
			id = int_val;
		}
	}
	Dictionary result = jsonrpc->make_response(p_result, id);
	memdelete(jsonrpc);
	return result;
}

Dictionary MCPProtocol::make_notification(const String &p_method, const Variant &p_params) {
	JSONRPC *jsonrpc = memnew(JSONRPC);
	Dictionary result = jsonrpc->make_notification(p_method, p_params);
	memdelete(jsonrpc);
	return result;
}

Dictionary MCPProtocol::make_progress_notification(const String &p_progress_token, const Dictionary &p_progress) {
	Dictionary notification;
	notification["jsonrpc"] = "2.0";
	notification["method"] = "notifications/progress";
	Dictionary params;
	params["progressToken"] = p_progress_token;
	params["progress"] = p_progress;
	notification["params"] = params;
	return notification;
}

bool MCPProtocol::is_batch_request(const Variant &p_message) {
	return p_message.get_type() == Variant::ARRAY;
}

bool MCPProtocol::validate_message(const Dictionary &p_message) {
	// Must have jsonrpc field set to "2.0"
	if (!p_message.has("jsonrpc")) {
		return false;
	}
	if (p_message["jsonrpc"] != "2.0") {
		return false;
	}

	// Must have either method (request/notification) or result/error (response)
	if (!p_message.has("method") && !p_message.has("result") && !p_message.has("error")) {
		return false;
	}

	return true;
}

Variant MCPProtocol::process_message(JSONRPC *p_jsonrpc, const Dictionary &p_message) {
	if (!p_jsonrpc) {
		return Variant();
	}
	return p_jsonrpc->process_action(p_message);
}

Array MCPProtocol::process_batch(JSONRPC *p_jsonrpc, const Array &p_messages) {
	Array results;
	if (!p_jsonrpc) {
		return results;
	}

	for (int i = 0; i < p_messages.size(); i++) {
		Variant message = p_messages[i];
		if (message.get_type() == Variant::DICTIONARY) {
			Dictionary msg_dict = message;
			Variant result = p_jsonrpc->process_action(msg_dict, true);
			if (result.get_type() != Variant::NIL) {
				results.push_back(result);
			}
		}
	}

	return results;
}
