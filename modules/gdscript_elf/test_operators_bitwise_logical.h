#ifndef TEST_OPERATORS_BITWISE_LOGICAL_H
#define TEST_OPERATORS_BITWISE_LOGICAL_H

#include "modules/gdscript/gdscript_compiler.h"
#include "modules/gdscript/gdscript_function.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript_elf/test_gdscript_c_generation.h"
#include "tests/test_macros.h"

// Bitwise and Logical operator tests - split for accessibility

namespace TestBitwiseLogicalOperators {

TEST_CASE("[GDScript][ELF][Operator] BITWISE SHIFT_LEFT") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_shl(a: int, b: int) -> int:
            return a << b
    )",
			"test_shl", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] BITWISE SHIFT_RIGHT") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_shr(a: int, b: int) -> int:
            return a >> b
    )",
			"test_shr", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] BITWISE BIT_AND") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_bitand(a: int, b: int) -> int:
            return a & b
    )",
			"test_bitand", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] BITWISE BIT_OR") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_bitor(a: int, b: int) -> int:
            return a | b
    )",
			"test_bitor", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] BITWISE BIT_XOR") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_bitxor(a: int, b: int) -> int:
            return a ^ b
    )",
			"test_bitxor", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] BITWISE BIT_NEGATE") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_bitnot(x: int) -> int:
            return ~x
    )",
			"test_bitnot", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] LOGICAL AND") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_log_and(a: bool, b: bool) -> bool:
            return a and b
    )",
			"test_log_and", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] LOGICAL OR") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_log_or(a: bool, b: bool) -> bool:
            return a or b
    )",
			"test_log_or", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] LOGICAL NOT") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_log_not(x: bool) -> bool:
            return not x
    )",
			"test_log_not", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] LOGICAL XOR") {
	// Note: GDScript doesn't have direct XOR, but we can test != for bool XOR
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_bool_xor(a: bool, b: bool) -> bool:
            return a != b
    )",
			"test_bool_xor", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] CONTAINMENT IN") {
	TestGDScriptCGeneration::test_operator_validation(R"(
        func test_in_operator(item: int) -> bool:
            return item in [1, 2, 3, 4, 5]
    )",
			"test_in_operator", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] Complex bitwise expression") {
	const String bitwise_ops = R"(
        func test_bitwise_expr(x: int, y: int, z: int) -> int:
            var mask = x & y
            var shifted = mask << 2
            var inverted = ~shifted
            return inverted ^ z
    )";

	GDScriptFunction *func = TestGDScriptCGeneration::create_and_find_function(bitwise_ops, "test_bitwise_expr");
	String c_code = TestGDScriptCGeneration::generate_and_verify_c_code(func);

	// Should contain BIT_AND, SHIFT_LEFT, BIT_NEGATE, BIT_XOR operations
	REQUIRE(c_code.contains("operator_funcs["));
	int bitwise_ops_count = 0;
	int pos = 0;
	while ((pos = c_code.find("operator_funcs[", pos)) != -1) {
		bitwise_ops_count++;
		pos += 14;
	}
	REQUIRE(bitwise_ops_count >= 3); // Should have BIT_AND, SHIFT_LEFT, BIT_XOR (+ BIT_NEGATE potentially)
}

TEST_CASE("[GDScript][ELF][Operator] Complex logical expression") {
	const String logical_ops = R"(
        func test_logical_chain(a: bool, b: bool, c: bool) -> bool:
            return (a and b) or (not c)
    )";

	GDScriptFunction *func = TestGDScriptCGeneration::create_and_find_function(logical_ops, "test_logical_chain");
	String c_code = TestGDScriptCGeneration::generate_and_verify_c_code(func);

	// Should contain logical operations
	REQUIRE(c_code.contains("operator_funcs["));
}

TEST_CASE("[GDScript][ELF][Operator] Mixed bitwise and arithmetic") {
	const String mixed_ops = R"(
        func test_mixed_ops(x: int, shift: int, mask: int) -> int:
            var shifted = x << shift
            var masked = shifted & mask
            return masked + (x >> 1)
    )";

	GDScriptFunction *func = TestGDScriptCGeneration::create_and_find_function(mixed_ops, "test_mixed_ops");
	String c_code = TestGDScriptCGeneration::generate_and_verify_c_code(func);

	// Should contain both bitwise and arithmetic operations
	REQUIRE(c_code.contains("operator_funcs["));
	int total_ops = 0;
	int pos = 0;
	while ((pos = c_code.find("operator_funcs[", pos)) != -1) {
		total_ops++;
		pos += 14;
	}
	REQUIRE(total_ops >= 4); // SHIFT_LEFT, BIT_AND, SHIFT_RIGHT, ADD
}

} // namespace TestBitwiseLogicalOperators

#endif // TEST_OPERATORS_BITWISE_LOGICAL_H
