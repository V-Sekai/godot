/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "http_stream_transport.h"
#include "mcp_protocol.h"
#include "mcp_server.h"
#include "script_server.h"
#include "mcp_runner.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "core/object/object.h"

static MCPServer *mcp_server_singleton = nullptr;
static Ref<MCPScriptServer> script_server_singleton;
static MCPRunner *mcp_runner_node = nullptr;

void initialize_mcp_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	GDREGISTER_CLASS(MCPServer);
	GDREGISTER_CLASS(MCPScriptServer);
	GDREGISTER_CLASS(MCPRunner);

	// Create and register singletons
	mcp_server_singleton = memnew(MCPServer);
	Engine::get_singleton()->add_singleton(Engine::Singleton("MCPServer", mcp_server_singleton));

	script_server_singleton.instantiate();
	Engine::get_singleton()->add_singleton(Engine::Singleton("MCPScriptServer", script_server_singleton.ptr()));

	// Create runner and add to scene tree via MainLoop if possible, 
	// but in editor we might need a different approach.
	// For now, let's just create it.
	mcp_runner_node = memnew(MCPRunner);
	// We'll add it to the root node when it becomes available
}

void uninitialize_mcp_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	if (mcp_runner_node) {
		memdelete(mcp_runner_node);
		mcp_runner_node = nullptr;
	}

	// Cleanup singletons
	if (Engine::get_singleton()->has_singleton("MCPScriptServer")) {
		Engine::get_singleton()->remove_singleton("MCPScriptServer");
		script_server_singleton.unref();
	}

	if (Engine::get_singleton()->has_singleton("MCPServer")) {
		Engine::get_singleton()->remove_singleton("MCPServer");
		if (mcp_server_singleton) {
			memdelete(mcp_server_singleton);
			mcp_server_singleton = nullptr;
		}
	}
}
