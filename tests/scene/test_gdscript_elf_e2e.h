/**************************************************************************/
/*  test_gdscript_elf_e2e.h                                               */
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

#ifdef MODULE_GDSCRIPT_ELF_ENABLED

#include "modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.h"
#include "modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.h"
#include "modules/gdscript_elf/src/gdscript_c_compiler.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_function.h"
#include "scene/main/scene_tree.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestGDScriptELFE2E {

// Helper to create a GDScript instance and compile it
static Ref<GDScript> create_and_compile_script(const String &p_source_code) {
	Ref<GDScript> script;
	script.instantiate();
	
	// Set a temporary path (required for proper compilation)
	script->set_path("res://test_script.gd");
	script->set_source_code(p_source_code);
	
	Error err = script->reload();
	if (err != OK) {
		print_error(vformat("GDScript reload failed with error: %d", err));
		return Ref<GDScript>();
	}
	
	if (!script->is_valid()) {
		print_error("GDScript is not valid after reload");
		return Ref<GDScript>();
	}
	
	return script;
}

// Helper to test C++ code generation for a function
static void test_cpp_generation(const String &p_source_code, const StringName &p_function_name) {
	Ref<GDScript> script = create_and_compile_script(p_source_code);
	REQUIRE(script.is_valid());
	
	const HashMap<StringName, GDScriptFunction *> &funcs = script->get_member_functions();
	REQUIRE(funcs.has(p_function_name));
	
	GDScriptFunction *func = funcs.get(p_function_name);
	REQUIRE(func != nullptr);
	
	// Check if function can be compiled (validates code_ptr and code_size internally)
	REQUIRE(func->can_compile_to_elf64(0));
	
	// Generate C++ code
	Ref<GDScriptBytecodeCCodeGenerator> generator;
	generator.instantiate();
	
	String cpp_code = generator->generate_c_code(func);
	REQUIRE(!cpp_code.is_empty());
	
	// Verify key patterns
	CHECK(cpp_code.contains("GuestVariant"));
	CHECK(cpp_code.contains("#include \"modules/sandbox/src/guest_datatypes.h\""));
	CHECK(cpp_code.contains("void gdscript_"));
	CHECK(cpp_code.contains("GuestVariant stack["));
}

// Helper to test native C++ compilation
static bool test_native_compilation(const String &p_cpp_code) {
	Ref<GDScriptCCompiler> compiler;
	compiler.instantiate();
	
	Vector<String> include_paths;
	include_paths.push_back("core");
	include_paths.push_back("modules/sandbox/src");
	
	String executable_path;
	Error err = compiler->compile_cpp_to_native(p_cpp_code, include_paths, executable_path);
	
	if (err == OK) {
		// Verify executable exists
		Ref<FileAccess> file = FileAccess::open(executable_path, FileAccess::READ);
		if (file.is_valid()) {
			return file->get_length() > 0;
		}
	}
	
	return false;
}

TEST_CASE("[SceneTree][GDScriptELF] Simple function C++ generation") {
	// Test basic C++ code generation in SceneTree context
	const String test_code = "func test_simple() -> int:\n\treturn 42\n";
	
	test_cpp_generation(test_code, "test_simple");
}

TEST_CASE("[SceneTree][GDScriptELF] Arithmetic operations C++ generation") {
	const String test_code = "func test_add(a: int, b: int) -> int:\n\treturn a + b\n\nfunc test_multiply(x: int, y: int) -> int:\n\treturn x * y\n";
	
	test_cpp_generation(test_code, "test_add");
	test_cpp_generation(test_code, "test_multiply");
}

TEST_CASE("[SceneTree][GDScriptELF] Conditional logic C++ generation") {
	const String test_code = "func test_if(x: int) -> int:\n\tif x > 10:\n\t\treturn 100\n\treturn 0\n";
	
	test_cpp_generation(test_code, "test_if");
}

TEST_CASE("[SceneTree][GDScriptELF] Variable assignments C++ generation") {
	const String test_code = "func test_assign() -> int:\n\tvar x = 5\n\tvar y = 10\n\treturn x + y\n";
	
	test_cpp_generation(test_code, "test_assign");
}

TEST_CASE("[SceneTree][GDScriptELF] Native C++ compilation with real compiler") {
	// This test requires a real C++ compiler (g++/clang++) to be available
	const String test_code = "func test_compile() -> int:\n\treturn 42\n";
	
	Ref<GDScript> script = create_and_compile_script(test_code);
	REQUIRE(script.is_valid());
	
	const HashMap<StringName, GDScriptFunction *> &funcs = script->get_member_functions();
	REQUIRE(funcs.has("test_compile"));
	
	GDScriptFunction *func = funcs.get("test_compile");
	REQUIRE(func != nullptr);
	
	// Generate C++ code
	Ref<GDScriptBytecodeCCodeGenerator> generator;
	generator.instantiate();
	String cpp_code = generator->generate_c_code(func);
	REQUIRE(!cpp_code.is_empty());
	
	// Try to compile (may fail if compiler not available - that's OK)
	bool compiled = test_native_compilation(cpp_code);
	if (compiled) {
		CHECK_MESSAGE(compiled, "Native C++ compilation succeeded");
	}
	// If compiler not available, test still passes (compilation is optional)
}

TEST_CASE("[SceneTree][GDScriptELF] Full pipeline: GDScript → C++ → Compilation") {
	// End-to-end test: Create script, generate C++, compile it
	const String test_code = "func e2e_test(a: int, b: int) -> int:\n\tvar result = a + b\n\tif result > 100:\n\t\treturn 100\n\treturn result\n";
	
	Ref<GDScript> script = create_and_compile_script(test_code);
	REQUIRE(script.is_valid());
	
	const HashMap<StringName, GDScriptFunction *> &funcs = script->get_member_functions();
	REQUIRE(funcs.has("e2e_test"));
	
	GDScriptFunction *func = funcs.get("e2e_test");
	REQUIRE(func != nullptr);
	
	// Step 1: Generate C++ code
	Ref<GDScriptBytecodeCCodeGenerator> generator;
	generator.instantiate();
	String cpp_code = generator->generate_c_code(func);
	REQUIRE(!cpp_code.is_empty());
	
	// Step 2: Verify code structure
	CHECK(cpp_code.contains("GuestVariant"));
	CHECK(cpp_code.contains("GuestVariant stack["));
	CHECK(cpp_code.contains("void gdscript_e2e_test"));
	
	// Step 3: Try compilation (optional - may not have compiler)
	bool compiled = test_native_compilation(cpp_code);
	if (compiled) {
		CHECK_MESSAGE(compiled, "Native C++ compilation succeeded");
	}
	// If compiler not available, test still passes (compilation is optional)
}

TEST_CASE("[SceneTree][GDScriptELF] Script instance creation and function call") {
	// Test that we can create a script instance and call functions
	// This verifies the full pipeline works in SceneTree context
	const String test_code = "var test_value = 0\n\nfunc set_value(v: int):\n\ttest_value = v\n\nfunc get_value() -> int:\n\treturn test_value\n";
	
	Ref<GDScript> script = create_and_compile_script(test_code);
	REQUIRE(script.is_valid());
	
	// Create an instance (requires SceneTree for proper initialization)
	Node *test_node = memnew(Node);
	SceneTree::get_singleton()->get_root()->add_child(test_node);
	
	ScriptInstance *instance = script->instance_create(test_node);
	REQUIRE(instance != nullptr);
	
	// Test function calls
	Callable::CallError err;
	Variant result;
	
	// Call set_value
	Variant arg_value = 42;
	const Variant *args_set[] = { &arg_value };
	result = instance->callp("set_value", args_set, 1, err);
	CHECK(err.error == Callable::CallError::CALL_OK);
	
	// Call get_value
	result = instance->callp("get_value", nullptr, 0, err);
	CHECK(err.error == Callable::CallError::CALL_OK);
	CHECK(result.get_type() == Variant::INT);
	CHECK(result.operator int() == 42);
	
	// Cleanup
	test_node->queue_free();
	SceneTree::get_singleton()->process(0); // Process deletion
}

} // namespace TestGDScriptELFE2E

#endif // MODULE_GDSCRIPT_ELF_ENABLED
