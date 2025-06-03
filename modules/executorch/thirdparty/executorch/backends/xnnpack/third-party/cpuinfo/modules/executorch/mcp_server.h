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

#include "core/string/ustring.h"
#include "scene/main/node.h"

class ModelContextProtocolServer : public Node {
	GDCLASS(ModelContextProtocolServer, Node);

private:
	bool server_running;
	int port;
	String server_name;

	// MCP protocol state
	Dictionary capabilities;
	Array tools;
	Array resources;

protected:
	static void _bind_methods();

public:
	ModelContextProtocolServer();
	~ModelContextProtocolServer();

	// Node overrides
	void _ready();
	void _exit_tree();

	// MCP Server functionality
	void start_server(int p_port = 8080);
	void stop_server();
	bool is_server_running() const;

	// MCP Protocol methods
	void initialize_mcp();
	void add_tool(const String &name, const String &description, const Dictionary &schema);
	void add_resource(const String &uri, const String &name, const String &description);
	Dictionary handle_request(const Dictionary &request);

	// Property setters/getters
	void set_port(int p_port);
	int get_port() const;
	void set_server_name(const String &p_name);
	String get_server_name() const;

	// Signal callbacks
	void _on_client_connected();
	void _on_client_disconnected();
	void _on_message_received(const Dictionary &message);
};
