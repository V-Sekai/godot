/**************************************************************************/
/*  mcp_client.h                                                          */
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

#include "core/object/ref_counted.h"
#include "scene/main/http_request.h"

class MCPClient : public RefCounted {
	GDCLASS(MCPClient, RefCounted);

private:
	HTTPRequest *http_request;
	String server_url;
	bool connected;
	int instance_id;

	// JSON-RPC request ID counter
	int next_request_id;

	void _http_request_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _initialize_mcp();
	void _send_jsonrpc_request(const String &method, const Dictionary &params = Dictionary());

protected:
	static void _bind_methods();

public:
	MCPClient();
	~MCPClient();

	void connect_to_server(const String &url);
	void disconnect_from_server();
	bool is_connected() const { return connected; }

	// Godot instance management
	void create_instance(const String &project_path, const String &scene_path = "");
	void start_instance();
	void shutdown_instance();

	// Node operations
	void call_method(const String &node_path, const String &method_name, const Array &args = Array());
	Variant get_property(const String &node_path, const String &property_name);
	void set_property(const String &node_path, const String &property_name, const Variant &value);

	// Iteration/frame update
	void iteration();

	// Event handling
	void _notification(int p_what);
};
