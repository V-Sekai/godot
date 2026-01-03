/**************************************************************************/
/*  mcp_server.h                                                          */
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

#include "core/object/class_db.h"
#include "scene/main/node.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "http_stream_transport.h"
#include "mcp_protocol.h"
#include "modules/jsonrpc/jsonrpc.h"

class MCPServer : public Node {
	GDCLASS(MCPServer, Node);

private:
	Ref<HTTPStreamTransport> transport;
	JSONRPC *jsonrpc;
	HashMap<String, Callable> tool_handlers;
	HashMap<String, String> tool_descriptions;
	HashMap<String, Dictionary> tool_schemas;
	HashMap<String, Callable> resource_handlers;
	HashMap<Variant, int64_t> request_to_client; // Track which client made which request
	bool is_initialized = false; // Global initialization state for HTTP/SSE
	HashMap<String, Dictionary> prompt_definitions;
	Dictionary server_capabilities;
	String server_version;
	String server_name;

	// Auto-start functionality
	bool auto_start_enabled = false;
	mutable int auto_start_port = 8083;

	void _handle_message(int64_t p_client_id, const Variant &p_message);
	void _handle_initialize(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id);
	void _handle_tools_list(int64_t p_client_id, const Variant &p_request_id);
	void _handle_tools_call(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id);
	void _handle_resources_list(int64_t p_client_id, const Variant &p_request_id);
	void _handle_resources_read(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id);
	void _handle_prompts_list(int64_t p_client_id, const Variant &p_request_id);
	void _handle_prompts_get(int64_t p_client_id, const Dictionary &p_params, const Variant &p_request_id);

	Variant _handle_server_info_resource(const Dictionary &p_params, const Variant &p_request_id, int64_t p_client_id);

protected:
	static void _bind_methods();

public:
	MCPServer();
	~MCPServer();

	void start_server(int p_port);
	void stop_server();
	bool is_server_running() const;

	void register_tool(const String &p_name, const Variant &p_description_or_handler, const Callable &p_handler = Callable(), const Dictionary &p_schema = Dictionary());
	void register_resource(const String &p_name, const Callable &p_handler);
	void register_prompt(const String &p_name, const Dictionary &p_definition);
	void send_response(int64_t p_client_id, const Variant &p_request_id, const Variant &p_result);
	void send_notification(int64_t p_client_id, const String &p_method, const Dictionary &p_params);
	void send_202_accepted(int64_t p_client_id);
	void send_progress(int64_t p_client_id, const String &p_progress_token, const Dictionary &p_progress);

	void poll();

	// Auto-start functionality
	void check_auto_start();
	bool should_auto_start() const;
};
