/**************************************************************************/
/*  test_gdscript_to_c99.cpp                                             */
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
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tests/test_macros.h"

#include "../gdscript.h"
#include "../gdscript_to_c99.h"

#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "core/string/ustring.h"

namespace doctest {
template <typename T>
String toString(const T &in) {
	return String(in);
}
} // namespace doctest

TEST_CASE("[Modules][GDScript] C99 conversion on runtime test scripts") {
	// Test C99 conversion on official runtime test scripts from mlir/runtime_tests/features
	
	// List of official test files to test
	Vector<String> test_files = {
		"res://mlir/runtime_tests/features/argument_count.gd",
		"res://mlir/runtime_tests/features/array_string_stringname_equivalent.gd",
		"res://mlir/runtime_tests/features/abstract_methods.gd",
	};
	
	int total_tested = 0;
	int total_converted = 0;
	int total_failed = 0;
	
	for (const String &file_path : test_files) {
		// Load script
		Ref<GDScript> script = ResourceLoader::load(file_path);
		if (!script.is_valid()) {
			ERR_PRINT(vformat("Failed to load script: %s", file_path));
			total_failed++;
			continue;
		}
		
		// Reload to ensure it's compiled
		Error err = script->reload();
		if (err != OK) {
			ERR_PRINT(vformat("Failed to reload script: %s (error: %d)", file_path, err));
			total_failed++;
			continue;
		}
		
		// Get functions
		const HashMap<StringName, GDScriptFunction *> &functions = script->get_member_functions();
		
		for (const KeyValue<StringName, GDScriptFunction *> &E : functions) {
			StringName func_name = E.key;
			GDScriptFunction *func = E.value;
			
			if (!func) {
				continue;
			}
			
			total_tested++;
			
			// Check if can convert
			bool can_convert = GDScriptToC99::can_convert_to_c99(func);
			
			if (can_convert) {
				// Generate C99
				String c99_code = GDScriptToC99::generate_c99(func);
				
				if (c99_code.is_empty()) {
					ERR_PRINT(vformat("Failed to generate C99 for %s::%s", file_path, func_name));
					total_failed++;
				} else {
					total_converted++;
					
					// Log success (first few only to avoid spam)
					if (total_converted <= 3) {
						MESSAGE(vformat("✓ Converted %s::%s (%d bytes)", file_path, func_name, c99_code.length()));
					}
				}
			}
		}
	}
	
	// Summary
	MESSAGE(vformat("\nC99 Conversion Test Summary:"));
	MESSAGE(vformat("  Total functions tested: %d", total_tested));
	MESSAGE(vformat("  Successfully converted: %d", total_converted));
	MESSAGE(vformat("  Failed: %d", total_failed));
	
	if (total_tested > 0) {
		double success_rate = 100.0 * total_converted / total_tested;
		MESSAGE(vformat("  Success rate: %.1f%%", success_rate));
	}
	
	// Assert that we converted at least some functions
	CHECK(total_converted > 0);
}
