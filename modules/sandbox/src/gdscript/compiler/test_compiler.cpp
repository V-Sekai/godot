/**************************************************************************/
/*  test_compiler.cpp                                                     */
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
#include <fstream>
#include <iostream>
#include <string>

using namespace gdscript;

int main(int argc, char **argv) {
	std::cout << "GDScript to RISC-V Compiler Test" << std::endl;
	std::cout << "=================================" << std::endl
			  << std::endl;

	// Test program: simple function that adds two numbers
	std::string test_source = R"(
func add(a, b):
	return a + b

func test():
	return 123

func sum_to_n(n):
	return n

func main():
	var x = 10
	var y = 20
	return x + y
)";

	std::cout << "Source code:" << std::endl;
	std::cout << test_source << std::endl
			  << std::endl;

	// Compile with all debug output
	CompilerOptions options;
	options.dump_tokens = true;
	options.dump_ast = true;
	options.dump_ir = true;
	options.output_elf = true;
	options.output_path = "test_output.elf";

	Compiler compiler;
	auto elf_data = compiler.compile(test_source, options);

	if (elf_data.empty()) {
		std::cerr << "Compilation failed: " << compiler.get_error() << std::endl;
		return 1;
	}

	std::cout << "=== COMPILATION SUCCESS ===" << std::endl;
	std::cout << "Generated ELF size: " << elf_data.size() << " bytes" << std::endl;

	// Write to file
	if (compiler.compile_to_file(test_source, options.output_path, options)) {
		std::cout << "Output written to: " << options.output_path << std::endl;
	} else {
		std::cerr << "Failed to write output: " << compiler.get_error() << std::endl;
		return 1;
	}

	return 0;
}
