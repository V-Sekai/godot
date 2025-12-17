/**************************************************************************/
/*  test_branch_value_extraction.h                                       */
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
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFLICTING. */
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

namespace TestBranchValueExtraction {

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
	String script_path = "res://test_script.gd";
	Error err;
	Ref<FileAccess> file = FileAccess::open(script_path, FileAccess::WRITE, &err);
	if (err != OK || !file.is_valid()) {
		print_error(vformat("Failed to open file for writing: %s", script_path));
		return Ref<GDScript>();
	}

	file->store_string(p_source_code);
	file->close();

	Ref<GDScript> script = ResourceLoader::load(script_path, "GDScript", ResourceFormatLoader::CACHE_MODE_IGNORE);
	if (!script.is_valid()) {
		print_error(vformat("Failed to load script from: %s", script_path));
		return Ref<GDScript>();
	}

	return script;
}

// Test branch value extraction for a function
static void test_branch_extraction(const String &p_source_code, const StringName &p_function_name, bool p_should_succeed = true) {
	Ref<GDScript> script = create_and_compile_script(p_source_code);
	if (!script.is_valid() || !script->is_valid()) {
		if (!p_should_succeed) {
			return; // Expected to fail
		}
		FAIL("Script compilation failed");
		return;
	}

	const HashMap<StringName, GDScriptFunction *> &funcs = script->get_member_functions();
	if (!funcs.has(p_function_name)) {
		if (!p_should_succeed) {
			return; // Expected to fail
		}
		FAIL("Function not found");
		return;
	}

	GDScriptFunction *func = funcs.get(p_function_name);
	if (func == nullptr) {
		FAIL("Function is null");
		return;
	}

	// Test StableHLO conversion
	if (!GDScriptToStableHLO::can_convert_function(func)) {
		if (!p_should_succeed) {
			return; // Expected to fail
		}
		FAIL("Function cannot be converted");
		return;
	}

	String stablehlo_text = GDScriptToStableHLO::convert_function_to_stablehlo_text(func);
	if (stablehlo_text.is_empty()) {
		if (!p_should_succeed) {
			return; // Expected to fail
		}
		FAIL("StableHLO text is empty");
		return;
	}

	// Verify StableHLO contains expected patterns
	CHECK(stablehlo_text.contains("module"));
	CHECK(stablehlo_text.contains("func.func"));
	CHECK(stablehlo_text.contains("stablehlo"));

	// Check that placeholders (100.0, 0.0) are NOT used for branch values
	// (they should be replaced with actual extracted values)
	bool has_placeholder_100 = stablehlo_text.contains("100.0");

	// Note: 0.0 might be legitimate (e.g., for zero constants), but 100.0 should not appear
	// unless it's actually in the source code
	if (has_placeholder_100 && !p_source_code.contains("100")) {
		FAIL("Found placeholder 100.0 in StableHLO output - branch value extraction may have failed");
	}
}

TEST_CASE("[SceneTree][BranchValueExtraction] Simple conditional with constants") {
	init("branch_value_extraction");

	const String test_code = "func test_if(x: int) -> int:\n\tif x > 10:\n\t\treturn 100\n\treturn 0\n";

	test_branch_extraction(test_code, "test_if");
}

TEST_CASE("[SceneTree][BranchValueExtraction] Ternary operator pattern") {
	init("branch_value_extraction");

	const String test_code = "func test_ternary(x: int) -> int:\n\treturn 100 if x > 10 else 0\n";

	test_branch_extraction(test_code, "test_ternary");
}

TEST_CASE("[SceneTree][BranchValueExtraction] Conditional with arithmetic") {
	init("branch_value_extraction");

	const String test_code = "func test_arithmetic(x: int, y: int) -> int:\n\tif x > y:\n\t\treturn x + y\n\treturn x - y\n";

	test_branch_extraction(test_code, "test_arithmetic");
}

TEST_CASE("[SceneTree][BranchValueExtraction] Conditional with type test") {
	init("branch_value_extraction");

	const String test_code = "func test_type_test(x) -> int:\n\tif x is int:\n\t\treturn 1\n\treturn 0\n";

	test_branch_extraction(test_code, "test_type_test");
}

TEST_CASE("[SceneTree][BranchValueExtraction] Nested conditionals") {
	init("branch_value_extraction");

	const String test_code = "func test_nested(x: int, y: int) -> int:\n\tif x > 10:\n\t\tif y > 5:\n\t\t\treturn 100\n\t\treturn 50\n\treturn 0\n";

	test_branch_extraction(test_code, "test_nested");
}

TEST_CASE("[SceneTree][BranchValueExtraction] Complex expression chains") {
	init("branch_value_extraction");

	const String test_code = "func test_complex(x: int, y: int) -> int:\n\tif x * 2 > y + 10:\n\t\tvar result = x + y\n\t\treturn result * 2\n\treturn y - x\n";

	test_branch_extraction(test_code, "test_complex");
}

} // namespace TestBranchValueExtraction
