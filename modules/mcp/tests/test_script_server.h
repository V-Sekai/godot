/**************************************************************************/
/*  test_script_server.h                                                  */
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

#include "../script_server.h"

namespace TestMCPScriptServer {

TEST_CASE("[MCPScriptServer] Security checks - dangerous patterns") {
	Ref<MCPScriptServer> script_server = memnew(MCPScriptServer);

	// Test OS.execute (dangerous)
	String dangerous_code = "func run():\n    OS.execute('rm', ['-rf', '/'])";
	Dictionary result = script_server->_apply_security_checks(dangerous_code, "public");
	CHECK(result.has("error"));
	CHECK(String(result["error"]).contains("OS.execute"));

	// Test FileAccess.open (dangerous)
	String file_code = "func run():\n    var f = FileAccess.open('user://secret.txt', FileAccess.WRITE)";
	result = script_server->_apply_security_checks(file_code, "public");
	CHECK(result.has("error"));
	CHECK(String(result["error"]).contains("FileAccess.open"));

	// Test safe code
	String safe_code = "func run():\n    return 2 + 3";
	result = script_server->_apply_security_checks(safe_code, "public");
	CHECK(result.is_empty());
}

TEST_CASE("[MCPScriptServer] Method security checks") {
	Ref<MCPScriptServer> script_server = memnew(MCPScriptServer);

	// Test dangerous method
	Dictionary result = script_server->_apply_method_security_checks("execute", Array());
	CHECK(result.has("error"));

	// Test safe method
	result = script_server->_apply_method_security_checks("get_name", Array());
	CHECK(result.is_empty());
}

TEST_CASE("[MCPScriptServer] Class whitelist") {
	Ref<MCPScriptServer> script_server = memnew(MCPScriptServer);

	// Test whitelisted classes
	CHECK(script_server->_is_class_allowed("Vector3") == true);
	CHECK(script_server->_is_class_allowed("Node3D") == false); // Node3D is not in the whitelist I saw earlier
	CHECK(script_server->_is_class_allowed("MeshInstance3D") == true);
	CHECK(script_server->_is_class_allowed("ArrayMesh") == true);

	// Test non-whitelisted classes
	CHECK(script_server->_is_class_allowed("OS") == false);
	CHECK(script_server->_is_class_allowed("FileAccess") == false);
	CHECK(script_server->_is_class_allowed("DirAccess") == false);
}

TEST_CASE("[MCPScriptServer] Action sequence validation") {
	Ref<MCPScriptServer> script_server = memnew(MCPScriptServer);

	// Test missing actions
	Dictionary params;
	Dictionary result = script_server->_tool_execute_action_sequence(params, "1", "1");
	CHECK(result["success"] == false);
	CHECK(String(result["error"]).contains("actions"));

	// Test invalid actions type
	params["actions"] = "not an array";
	result = script_server->_tool_execute_action_sequence(params, "1", "1");
	CHECK(result["success"] == false);
	CHECK(String(result["error"]).contains("array"));
}

} // namespace TestMCPScriptServer
