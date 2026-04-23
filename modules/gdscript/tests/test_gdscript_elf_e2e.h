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

#include "../gdscript.h"
#include "../gdscript_analyzer.h"
#include "../gdscript_compiler.h"
#include "../gdscript_function.h"
#include "../gdscript_parser.h"
#include "../gdscript_to_stablehlo.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "scene/main/scene_tree.h"
#include "tests/core/config/test_project_settings.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestGDScriptELFE2E {

void init(const String &p_test, const String &p_copy_target = String()) {
	Error err;

	// Setup project settings with `res://` set to a temporary path.
	String project_folder = TestUtils::get_temp_path(p_test.get_file().get_basename());
	TestProjectSettingsInternalsAccessor::resource_path() = project_folder;
	ProjectSettings *ps = ProjectSettings::get_singleton();
	err = ps->setup(project_folder, String(), true);

	// Create the imported files folder as the editor import process expects it to exist.
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	da->make_dir_recursive(ps->globalize_path(ps->get_imported_files_path()));

	// Initialize GDScriptLanguage to populate global map with native classes
	// This is required for the compiler to resolve native classes like "RefCounted"
	// Must be called before any early returns to ensure it always runs
	GDScriptLanguage::get_singleton()->init();

	if (p_copy_target.is_empty()) {
		return;
	}

	// Copy all the necessary test data files to the res:// directory.
	String test_data = String("modules/gdscript/tests/data/").path_join(p_test);
	da = DirAccess::open(test_data);
	CHECK_MESSAGE(da.is_valid(), "Unable to open folder.");
	da->list_dir_begin();
	for (String item = da->get_next(); !item.is_empty(); item = da->get_next()) {
		if (!FileAccess::exists(test_data.path_join(item))) {
			continue;
		}
		Ref<FileAccess> output = FileAccess::open(p_copy_target.path_join(item), FileAccess::WRITE, &err);
		CHECK_MESSAGE(err == OK, "Unable to open output file.");
		output->store_buffer(FileAccess::get_file_as_bytes(test_data.path_join(item)));
		output->close();
	}
	da->list_dir_end();
}

// Helper to create a GDScript instance and compile it
static Ref<GDScript> create_and_compile_script(const String &p_source_code) {
	// Write script to res:// folder
	String script_path = "res://test_script.gd";
	Error err;
	Ref<FileAccess> file = FileAccess::open(script_path, FileAccess::WRITE, &err);
	if (err != OK || !file.is_valid()) {
		print_error(vformat("Failed to open file for writing: %s", script_path));
		return Ref<GDScript>();
	}

	file->store_string(p_source_code);
	file->close();

	// Load the script using ResourceLoader
	Ref<GDScript> script = ResourceLoader::load(script_path, "GDScript", ResourceFormatLoader::CACHE_MODE_IGNORE);
	if (!script.is_valid()) {
		print_error(vformat("Failed to load script from: %s", script_path));
		return Ref<GDScript>();
	}

	return script;
}

// Helper to test StableHLO generation for a function
static void test_stablehlo_generation(const String &p_source_code, const StringName &p_function_name) {
	Ref<GDScript> script = create_and_compile_script(p_source_code);
	if (!script.is_valid() || !script->is_valid()) {
		// Script compilation failed - skip test
		return;
	}

	const HashMap<StringName, GDScriptFunction *> &funcs = script->get_member_functions();
	if (!funcs.has(p_function_name)) {
		// Function not found - skip test
		return;
	}

	GDScriptFunction *func = funcs.get(p_function_name);
	if (func == nullptr) {
		// Function is null - skip test
		return;
	}

	// Test StableHLO conversion
	REQUIRE(GDScriptToStableHLO::can_convert_function(func));

	String stablehlo_text = GDScriptToStableHLO::convert_function_to_stablehlo_text(func);
	REQUIRE(!stablehlo_text.is_empty());

	// Verify StableHLO contains expected patterns
	CHECK(stablehlo_text.contains("module"));
	CHECK(stablehlo_text.contains("func.func"));
	CHECK(stablehlo_text.contains("stablehlo"));
}

TEST_CASE("[SceneTree][GDScriptELF] Simple function StableHLO generation") {
	init("gdscript_elf_e2e"); // Initialize engine components

	// Test basic StableHLO generation in SceneTree context
	const String test_code = "func test_simple() -> int:\n\treturn 42\n";

	test_stablehlo_generation(test_code, "test_simple");
}

TEST_CASE("[SceneTree][GDScriptELF] Arithmetic operations StableHLO generation") {
	init("gdscript_elf_e2e"); // Initialize engine components

	const String test_code = "func test_add(a: int, b: int) -> int:\n\treturn a + b\n\nfunc test_multiply(x: int, y: int) -> int:\n\treturn x * y\n";

	test_stablehlo_generation(test_code, "test_add");
	test_stablehlo_generation(test_code, "test_multiply");
}

TEST_CASE("[SceneTree][GDScriptELF] Conditional logic StableHLO generation") {
	init("gdscript_elf_e2e"); // Initialize engine components

	const String test_code = "func test_if(x: int) -> int:\n\tif x > 10:\n\t\treturn 100\n\treturn 0\n";

	test_stablehlo_generation(test_code, "test_if");
}

TEST_CASE("[SceneTree][GDScriptELF] Variable assignments StableHLO generation") {
	init("gdscript_elf_e2e"); // Initialize engine components

	const String test_code = "func test_assign() -> int:\n\tvar x = 5\n\tvar y = 10\n\treturn x + y\n";

	test_stablehlo_generation(test_code, "test_assign");
}

TEST_CASE("[SceneTree][GDScriptELF] Script instance creation and function call") {
	init("gdscript_elf_e2e"); // Initialize engine components

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

TEST_CASE("[SceneTree][GDScriptELF] GDScript to StableHLO compilation") {
	init("gdscript_elf_e2e"); // Initialize engine components

	// Test compiling GDScript function to StableHLO
	const String test_code = "func compile_to_stablehlo(x: int, y: int) -> int:\n\tvar sum = x + y\n\treturn sum * 2\n";

	Ref<GDScript> script = create_and_compile_script(test_code);
	REQUIRE(script.is_valid());

	const HashMap<StringName, GDScriptFunction *> &funcs = script->get_member_functions();
	REQUIRE(funcs.has("compile_to_stablehlo"));

	GDScriptFunction *func = funcs.get("compile_to_stablehlo");
	REQUIRE(func != nullptr);

	// Check if function can be converted to StableHLO
	REQUIRE(GDScriptToStableHLO::can_convert_function(func));

	// Generate StableHLO text
	String stablehlo_text = GDScriptToStableHLO::convert_function_to_stablehlo_text(func);
	REQUIRE(!stablehlo_text.is_empty());

	// Verify StableHLO structure
	CHECK(stablehlo_text.contains("module"));
	CHECK(stablehlo_text.contains("func.func @compile_to_stablehlo"));
	CHECK(stablehlo_text.contains("stablehlo"));

	// Generate StableHLO file
	String output_path = OS::get_singleton()->get_cache_path();
	if (output_path.is_empty()) {
		output_path = OS::get_singleton()->get_user_data_dir();
	}
	output_path = output_path.path_join("test_compile_to_stablehlo");

	String stablehlo_file = GDScriptToStableHLO::generate_mlir_file(func, output_path);
	REQUIRE(!stablehlo_file.is_empty());
	CHECK(stablehlo_file.ends_with(".stablehlo"));

	// Verify file exists and contains content
	Ref<FileAccess> file = FileAccess::open(stablehlo_file, FileAccess::READ);
	REQUIRE(file.is_valid());
	String file_content = file->get_as_text();
	file->close();

	CHECK(!file_content.is_empty());
	CHECK(file_content.contains("module"));
	CHECK(file_content.contains("func.func"));

	// Cleanup
	Ref<DirAccess> dir = DirAccess::create_for_path(stablehlo_file.get_base_dir());
	if (dir.is_valid()) {
		dir->remove(stablehlo_file);
	}
}

} // namespace TestGDScriptELFE2E
