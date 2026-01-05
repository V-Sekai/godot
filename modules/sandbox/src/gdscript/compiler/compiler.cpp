/**************************************************************************/
/*  compiler.cpp                                                          */
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

#include "compiler.h"
#include "codegen.h"
#include "elf_builder.h"
#include "ir_optimizer.h"
#include "lexer.h"
#include "parser.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace gdscript {

Compiler::Compiler() {}

std::vector<uint8_t> Compiler::compile(const std::string &source, const CompilerOptions &options) {
	try {
		// Step 1: Lexical analysis
		Lexer lexer(source);
		auto tokens = lexer.tokenize();

		if (options.dump_tokens) {
			std::cout << "=== TOKENS ===" << std::endl;
			for (const auto &token : tokens) {
				std::cout << token.to_string() << std::endl;
			}
			std::cout << std::endl;
		}

		// Step 2: Parsing
		Parser parser(tokens);
		Program program = parser.parse();

		if (options.dump_ast) {
			std::cout << "=== AST ===" << std::endl;
			std::cout << "Functions: " << program.functions.size() << std::endl;
			for (const auto &func : program.functions) {
				std::cout << "  func " << func.name << "(";
				for (size_t i = 0; i < func.parameters.size(); i++) {
					if (i > 0)
						std::cout << ", ";
					std::cout << func.parameters[i].name;
				}
				std::cout << "): " << func.body.size() << " statements" << std::endl;
			}
			std::cout << std::endl;
		}

		// Step 3: Code generation (AST -> IR)
		CodeGenerator codegen;
		IRProgram ir_program = codegen.generate(program);

		if (options.dump_ir) {
			std::cout << "=== IR (unoptimized) ===" << std::endl;
			for (const auto &func : ir_program.functions) {
				std::cout << "Function: " << func.name << std::endl;
				std::cout << "  Max registers: " << func.max_registers << std::endl;
				std::cout << "  Instructions:" << std::endl;
				for (const auto &instr : func.instructions) {
					std::cout << "    " << instr.to_string() << std::endl;
				}
				std::cout << std::endl;
			}

			std::cout << "String constants:" << std::endl;
			for (size_t i = 0; i < ir_program.string_constants.size(); i++) {
				std::cout << "  [" << i << "] \"" << ir_program.string_constants[i] << "\"" << std::endl;
			}
			std::cout << std::endl;
		}

		// Step 3.5: Optimize IR
		IROptimizer optimizer;
		optimizer.optimize(ir_program);

		if (options.dump_ir) {
			std::cout << "=== IR (optimized) ===" << std::endl;
			for (const auto &func : ir_program.functions) {
				std::cout << "Function: " << func.name << std::endl;
				std::cout << "  Max registers: " << func.max_registers << std::endl;
				std::cout << "  Instructions:" << std::endl;
				for (const auto &instr : func.instructions) {
					std::cout << "    " << instr.to_string() << std::endl;
				}
				std::cout << std::endl;
			}
		}

		// Step 4: RISC-V code generation (IR -> machine code)
		// For now, we'll create a minimal stub ELF
		std::vector<uint8_t> elf_data;

		if (options.output_elf) {
			ElfBuilder elf_builder;
			elf_data = elf_builder.build(ir_program);
		}

		m_error.clear();
		return elf_data;

	} catch (const std::exception &e) {
		m_error = e.what();
		return {};
	}
}

bool Compiler::compile_to_file(const std::string &source, const std::string &output_path, const CompilerOptions &options) {
	auto elf_data = compile(source, options);

	if (elf_data.empty()) {
		return false;
	}

	std::ofstream out(output_path, std::ios::binary);
	if (!out) {
		m_error = "Failed to open output file: " + output_path;
		return false;
	}

	out.write(reinterpret_cast<const char *>(elf_data.data()), elf_data.size());
	out.close();

	return out.good();
}

} // namespace gdscript
