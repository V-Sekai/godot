/**************************************************************************/
/*  gdscript_elf64_writer.cpp                                             */
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

#include "gdscript_elf64_writer.h"
#include "gdscript_elf64_mode.h"

#include "core/string/print_string.h"
#include "gdscript_bytecode_elf_compiler.h"
#include "modules/gdscript/gdscript_function.h"

PackedByteArray GDScriptELF64Writer::write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode) {
	if (!p_function || p_function->_code_ptr == nullptr || p_function->_code_size == 0) {
		print_error("GDScriptELF64Writer: Invalid function or missing bytecode");
		return PackedByteArray();
	}

	// Use new C code generation + cross-compilation pipeline
	Ref<GDScriptBytecodeELFCompiler> compiler;
	compiler.instantiate();

	// Set up include paths for Godot and sandbox headers
	Vector<String> include_paths;
	include_paths.push_back("core/variant"); // For Variant type
	include_paths.push_back("modules/sandbox/src"); // For syscall numbers
	compiler->set_include_paths(include_paths);

	PackedByteArray elf_data;
	Error err = compiler->compile_function_to_elf64(p_function, elf_data);

	if (err != OK) {
		print_error(vformat("GDScriptELF64Writer: Failed to compile function '%s' to ELF (%s)",
				p_function->get_name(), error_names[err]));
		return PackedByteArray();
	}

	return elf_data;
}

bool GDScriptELF64Writer::can_write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode) {
	if (!p_function) {
		return false;
	}

	// Check if function has bytecode
	if (p_function->_code_ptr == nullptr || p_function->_code_size == 0) {
		return false;
	}

	// Check if C compilation pipeline can handle this function
	Ref<GDScriptBytecodeELFCompiler> compiler;
	compiler.instantiate();

	return compiler->can_compile_function_to_elf64(p_function);
}
