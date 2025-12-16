#ifndef TEST_OPERATORS_ARITHMETIC_H
#define TEST_OPERATORS_ARITHMETIC_H

#include "tests/test_macros.h"
#include "modules/gdscript/gdscript_compiler.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript/gdscript_function.h"
#include "modules/gdscript_elf/test_gdscript_c_generation.h"

// Arithmetic operator tests - split from main file for accessibility

namespace TestArithmeticOperators {

TEST_CASE("[GDScript][ELF][Operator] ADD - Integer addition") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_add_int(a: int, b: int) -> int:
            return a + b
    )", "test_add_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] ADD - Float addition") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_add_float(a: float, b: float) -> float:
            return a + b
    )", "test_add_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] SUBTRACT - Integer subtraction") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_sub_int(a: int, b: int) -> int:
            return a - b
    )", "test_sub_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] MULTIPLY - Float multiplication") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_mul_float(a: float, b: float) -> float:
            return a * b
    )", "test_mul_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] DIVIDE - Float division") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_div_float(a: float, b: float) -> float:
            return a / b
    )", "test_div_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] MODULE - Integer modulo") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_mod_int(a: int, b: int) -> int:
            return a % b
    )", "test_mod_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] POWER - Float power") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_pow_float(a: float, b: float) -> float:
            return a ** b
    )", "test_pow_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] NEGATE - Integer negation") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_neg_int(x: int) -> int:
            return -x
    )", "test_neg_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] NEGATE - Float negation") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_neg_float(x: float) -> float:
            return -x
    )", "test_neg_float", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] POSITIVE - Unary plus") {
    TestGDScriptCGeneration::test_operator_validation(R"(
        func test_pos_int(x: int) -> int:
            return +x
    )", "test_pos_int", "operator_funcs[");
}

TEST_CASE("[GDScript][ELF][Operator] Complex arithmetic expression") {
    const String complex_expr = R"(
        func test_complex_math(x: int, y: int, z: float) -> float:
            var result = (x + y) * z - (x / 2)
            return result + z ** 0.5
    )";

    GDScriptFunction* func = TestGDScriptCGeneration::create_and_find_function(complex_expr, "test_complex_math");
    String c_code = TestGDScriptCGeneration::generate_and_verify_c_code(func);

    // Should contain multiple validated operator calls for different operations
    REQUIRE(c_code.contains("operator_funcs["));
    // Should have ADD, MULTIPLY, SUBTRACT, DIVIDE, POWER operations
    int op_count = 0;
    int pos = 0;
    while ((pos = c_code.find("operator_funcs[", pos)) != -1) {
        op_count++;
        pos += 14;
    }
    REQUIRE(op_count >= 4); // Should have at least ADD, MULTIPLY, SUBTRACT, DIVIDE
}

} // namespace TestArithmeticOperators

#endif // TEST_OPERATORS_ARITHMETIC_H
