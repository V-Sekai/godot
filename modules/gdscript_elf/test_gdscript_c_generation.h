/**************************************************************************/
/*  test_gdscript_c_generation.h                                         */
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

#ifndef TEST_GDSCRIPT_C_GENERATION_H
#define TEST_GDSCRIPT_C_GENERATION_H

#include "tests/test_macros.h"
#include "modules/gdscript/gdscript_compiler.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript/gdscript_function.h"
#include "gdscript_bytecode_c_code_generator.h"
#include "gdscript_c_compiler.h"
#include "gdscript_bytecode_elf_compiler.h"

namespace TestGDScriptCGeneration {
    TEST_CASE("[GDScript][ELF][CGeneration] Basic code generation") {
        const String test_code = R"(
            func test_function(x: int, y: int) -> int:
                var result = x + y
                return result
        )";

        // Parse the GDScript code
        GDScriptParser parser;
        parser.set_source_code(test_code);

        Error parse_err = parser.parse();
        REQUIRE(parse_err == OK);

        // Generate bytecode for a function
        GDScriptCompiler compiler;
        Ref<GDScript> script = memnew(GDScript);
        script->set_source_code(test_code);

        GDScriptFunction *test_func = nullptr;

        // Find the test function
        for (const KeyValue<StringName, GDScriptFunction *> &E : script->get_member_functions()) {
            if (E.key == StringName("test_function")) {
                test_func = E.value;
                break;
            }
        }

        REQUIRE(test_func != nullptr);
        REQUIRE(test_func->_code_ptr != nullptr);
        REQUIRE(test_func->_code_size > 0);

        // Test C code generation
        Ref<GDScriptBytecodeCCodeGenerator> generator;
        generator.instantiate();

        String c_code = generator->generate_c_code(test_func);
        REQUIRE(!c_code.is_empty());

        print_verbose("Generated C code:");
        print_verbose(c_code);

        // Check that the generated code contains expected elements
        REQUIRE(c_code.contains("void gdscript_test_function"));
        REQUIRE(c_code.contains("Variant stack["));
        REQUIRE(c_code.contains("*result ="));
        REQUIRE(c_code.contains("return;"));
    }

    TEST_CASE("[GDScript][ELF][CGeneration] Opcode translation") {
        // Test specific opcode translations
        const String test_code = R"(
            func test_opcodes() -> int:
                var x = true
                var y = false
                var z = 42
                return z
        )";

        // Parse and get function (similar to above test)
        GDScriptParser parser;
        parser.set_source_code(test_code);
        Error parse_err = parser.parse();
        REQUIRE(parse_err == OK);

        Ref<GDScript> script = memnew(GDScript);
        script->set_source_code(test_code);

        GDScriptFunction *test_func = nullptr;
        for (const KeyValue<StringName, GDScriptFunction *> &E : script->get_member_functions()) {
            if (E.key == StringName("test_opcodes")) {
                test_func = E.value;
                break;
            }
        }

        REQUIRE(test_func != nullptr);

        Ref<GDScriptBytecodeCCodeGenerator> generator;
        generator.instantiate();

        String c_code = generator->generate_c_code(test_func);
        REQUIRE(!c_code.is_empty());

        // Check for expected assignments
        REQUIRE(c_code.contains(" = true;"));
        REQUIRE(c_code.contains(" = false;"));
        REQUIRE(c_code.contains(" = 42;"));
    }

    TEST_CASE("[GDScript][ELF][CCompiler] Cross-compiler detection") {
        Ref<GDScriptCCompiler> compiler;
        compiler.instantiate();

        // This test will pass if RISC-V cross-compiler is available,
        // or gracefully fail if not (which is expected in test environment)
        bool available = compiler->is_cross_compiler_available();

        if (available) {
            String compiler_path = compiler->detect_cross_compiler();
            REQUIRE(!compiler_path.is_empty());
            print_verbose("Cross-compiler found: " + compiler_path);
        } else {
            print_verbose("No cross-compiler found - this is expected in most test environments");
        }
    }
}

#endif // TEST_GDSCRIPT_C_GENERATION_H
