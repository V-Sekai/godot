/**************************************************************************/
/*  test_operators_comparison.h                                           */
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

#include "modules/gdscript/gdscript_compiler.h"
#include "modules/gdscript/gdscript_function.h"
#include "modules/gdscript/gdscript_parser.h"
#include "test_gdscript_c_generation.h"
#include "tests/test_macros.h"

// Comparison operator tests - split for accessibility

namespace TestComparisonOperators {

TEST_CASE("[GDScript][ELF][Operator] EQUAL - Integer equality") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_eq_int(a: int, b: int) -> bool:
            return a == b
    )",
			"test_eq_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] EQUAL - String equality") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_eq_str(a: String, b: String) -> bool:
            return a == b
    )",
			"test_eq_str", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] NOT_EQUAL - Integer inequality") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_ne_int(a: int, b: int) -> bool:
            return a != b
    )",
			"test_ne_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] NOT_EQUAL - Float inequality") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_ne_float(a: float, b: float) -> bool:
            return a != b
    )",
			"test_ne_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] LESS - Integer comparison") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_lt_int(a: int, b: int) -> bool:
            return a < b
    )",
			"test_lt_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] LESS - Float comparison") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_lt_float(a: float, b: float) -> bool:
            return a < b
    )",
			"test_lt_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] LESS_EQUAL - Integer comparison") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_le_int(a: int, b: int) -> bool:
            return a <= b
    )",
			"test_le_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] GREATER - Integer comparison") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_gt_int(a: int, b: int) -> bool:
            return a > b
    )",
			"test_gt_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] GREATER_EQUAL - Float comparison") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_ge_float(a: float, b: float) -> bool:
            return a >= b
    )",
			"test_ge_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] Complex comparison expression") {
	const String comparison_chain = R"(
        func test_comparison_chain(x: int, y: int, z: int) -> bool:
            return x < y and y <= z and x != z
    )";

	GDScriptFunction *func = TestGDScriptCGeneration::create_and_find_function(comparison_chain, "test_comparison_chain");
	String c_code = TestGDScriptCGeneration::generate_and_verify_c_code(func);

	// Should contain multiple comparison operators
	REQUIRE(c_code.contains("operator_funcs["));
	// Should test LESS, LESS_EQUAL, and NOT_EQUAL operations
	int compare_ops = 0;
	int pos = 0;
	while ((pos = c_code.find("operator_funcs[", pos)) != -1) {
		compare_ops++;
		pos += 14;
	}
	REQUIRE(compare_ops >= 2); // At least LESS and NOT_EQUAL
}

TEST_CASE("[GDScript][ELF][Operator] Mixed types - Int vs Float comparison") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_int_float_compare(i: int, f: float) -> bool:
            return i < f
    )",
			"test_int_float_compare", "operator_funcs[");
}

} // namespace TestComparisonOperators
