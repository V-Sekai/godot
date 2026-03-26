/**************************************************************************/
/*  scenetree_mcp.h                                                       */
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

#include "core/io/stream_peer_tcp.h"
#include "core/io/tcp_server.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "scene/main/scene_tree.h"

class SceneTreeMCP : public SceneTree {
	GDCLASS(SceneTreeMCP, SceneTree);

protected:
	static void _bind_methods();

public:
	struct RegisteredTool {
		String name;
		String description;
		Dictionary schema;
		Callable callable;
	};

	struct RegisteredResource {
		String uri;
		String name;
		Callable callable;
	};

	struct RegisteredPrompt {
		String name;
		String description;
		Array arguments;
		String template_text;
	};

	struct MCPServerPeer {
		Ref<StreamPeerTCP> peer;
		String request_buffer;
		bool is_http = false;
	};

	SceneTreeMCP();
	~SceneTreeMCP();

	// Public API for GDScript interop
	Error start_server(int p_port = 8765);
	void stop_server();

	// HTTP transport control
	Error start_http_server(int p_port = 8765);
	void stop_http_server();
	bool is_http_server_running() const { return http_server_running; }

	Error register_tool(const String &p_name, const String &p_description,
			const Dictionary &p_schema, const Callable &p_callable);
	Error register_resource(const String &p_uri, const String &p_name,
			const Callable &p_callable);

	void set_server_info(const String &p_name, const String &p_version);
	String get_server_name() const { return server_info_name; }
	String get_server_version() const { return server_info_version; }

	bool is_server_running() const { return server_running; }

	// Process server manually (if not using as main loop)
	void process_server(double p_delta);

private:
	// MainLoop lifecycle callbacks (automatically called by Godot)
	void initialize() override;
	bool process(double p_delta) override;
	void finalize() override;

	// Transport management
	void _start_transport(int p_port);
	void _stop_transport();
	void _accept_connections();
	bool _is_http_peer(const MCPServerPeer &p_peer);
	void _process_peers();
	void _prune_dead_peers();

	// HTTP parsing
	String _parse_http_request(const String &p_data, String &r_method,
			String &r_path, Dictionary &r_headers);
	String _format_http_response(int p_status_code, const String &p_status_text,
			const String &p_body, const Dictionary &p_headers = Dictionary());
	String _get_content_type(const Dictionary &p_headers);
	bool _is_streaming_request(const Dictionary &p_headers);
	String _format_jsonl_response(const Variant &p_response, const Variant &p_id);

	// JSON-RPC/MCP handlers
	Variant _handle_jsonrpc_request(const Variant &p_request, bool &r_should_reply);
	Variant _handle_jsonrpc_batch(const Array &p_batch, bool &r_should_reply);
	Dictionary _handle_mcp_initialize(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_initialized(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_tools_list(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_tools_call(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_resources_list(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_resources_read(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_prompts_list(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_prompts_get(const Dictionary &p_request, const Variant &p_id);
	Dictionary _handle_mcp_ping(const Dictionary &p_request, const Variant &p_id);

	Dictionary _make_jsonrpc_error(int p_code, const String &p_message, const Variant &p_id);
	Dictionary _make_jsonrpc_result(const Variant &p_result, const Variant &p_id);
	bool _is_supported_protocol_version(const String &p_version) const;

	void _register_builtin_method_tools();
	void _register_builtin_prompts();
	void _register_class_methods_as_tools(const StringName &p_class_name, const String &p_prefix);
	void _register_classdb_methods_as_tools();

	// ClassDB method wrappers
	Variant classdb_instantiate(const Array &p_args);
	Variant classdb_can_instantiate(const Array &p_args);
	Variant classdb_get_method_list(const Array &p_args);
	Variant classdb_class_exists(const Array &p_args);

	// SceneTree method wrappers
	Variant scenetree_get_root(const Array &p_args);

	Dictionary _build_generic_method_schema() const;
	void _collect_project_filesystem_entries(const String &p_dir, PackedStringArray &r_entries) const;
	bool _is_likely_binary_file(const String &p_path) const;
	String _read_project_filesystem_resource();

	// Server state
	Ref<TCPServer> tcp_server;
	Ref<TCPServer> http_server;
	LocalVector<MCPServerPeer> peers;
	bool server_running = false;
	bool http_server_running = false;
	int server_port = 8765;
	int http_server_port = 8765;

	// MCP state
	bool is_initialized = false;
	String negotiated_protocol_version = "2025-03-26";
	String server_info_name = "godot-mcp-server";
	String server_info_version = "1.0.0";

	// Registries
	HashMap<String, RegisteredTool> tool_registry;
	HashMap<String, RegisteredResource> resource_registry;
	HashMap<String, RegisteredPrompt> prompt_registry;
};
