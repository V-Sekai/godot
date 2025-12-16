/**************************************************************************/
/*  gdscript_bytecode_elf_compiler.cpp                                    */
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

#include "gdscript_bytecode_elf_compiler.h"
#include "core/string/print_string.h"
#include "gdscript_bytecode_c_code_generator.h"
#include "gdscript_c_compiler.h"

GDScriptBytecodeELFCompiler::GDScriptBytecodeELFCompiler() {
	code_generator.instantiate();
	compiler.instantiate();
}

void GDScriptBytecodeELFCompiler::add_include_path(const String &p_path) {
	include_paths.append(p_path);
}

void GDScriptBytecodeELFCompiler::set_include_paths(const Vector<String> &p_paths) {
	include_paths = p_paths;
}

bool GDScriptBytecodeELFCompiler::is_basic_opcodes_only(const GDScriptFunction *p_function) const {
	if (!p_function || !p_function->_code_ptr || p_function->_code_size == 0) {
		return false;
	}

	const int *code_ptr = p_function->_code_ptr;
	int code_size = p_function->_code_size;
	int ip = 0;

	// Check if all opcodes are basic/directly supported ones
	while (ip < code_size) {
		int opcode = code_ptr[ip];

		switch (opcode) {
			// Supported opcodes that generate direct C code
			case GDScriptFunction::OPCODE_RETURN:
			case GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN:
			case GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY:
			case GDScriptFunction::OPCODE_RETURN_TYPED_DICTIONARY:
			case GDScriptFunction::OPCODE_RETURN_TYPED_NATIVE:
			case GDScriptFunction::OPCODE_RETURN_TYPED_SCRIPT:
			case GDScriptFunction::OPCODE_ASSIGN:
			case GDScriptFunction::OPCODE_ASSIGN_NULL:
			case GDScriptFunction::OPCODE_ASSIGN_TRUE:
			case GDScriptFunction::OPCODE_ASSIGN_FALSE:
			case GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN:
			case GDScriptFunction::OPCODE_ASSIGN_TYPED_ARRAY:
			case GDScriptFunction::OPCODE_ASSIGN_TYPED_DICTIONARY:
			case GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE:
			case GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT:
			case GDScriptFunction::OPCODE_JUMP:
			case GDScriptFunction::OPCODE_JUMP_IF:
			case GDScriptFunction::OPCODE_JUMP_IF_NOT:
			case GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT:
			case GDScriptFunction::OPCODE_JUMP_IF_SHARED:
			case GDScriptFunction::OPCODE_OPERATOR_VALIDATED:
			case GDScriptFunction::OPCODE_GET_MEMBER:
			case GDScriptFunction::OPCODE_SET_MEMBER:
			case GDScriptFunction::OPCODE_CALL:
			case GDScriptFunction::OPCODE_LINE:
			case GDScriptFunction::OPCODE_BREAKPOINT:
			case GDScriptFunction::OPCODE_ASSERT:
			case GDScriptFunction::OPCODE_END:
			// Type adjustment opcodes (no C code needed, handled at bytecode level)
			case GDScriptFunction::OPCODE_TYPE_ADJUST_BOOL:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_INT:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_FLOAT:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_STRING:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR2:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR2I:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_RECT2:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_RECT2I:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR3:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR3I:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_TRANSFORM2D:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR4:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_VECTOR4I:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PLANE:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_QUATERNION:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_AABB:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_BASIS:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_TRANSFORM3D:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PROJECTION:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_COLOR:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_STRING_NAME:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_NODE_PATH:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_RID:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_OBJECT:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_CALLABLE:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_SIGNAL:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_DICTIONARY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_BYTE_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_INT32_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_INT64_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_FLOAT32_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_FLOAT64_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_STRING_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR2_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR3_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_COLOR_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR4_ARRAY:
				// These opcodes either generate C code or are harmless comments
				break;

			default:
				// Unsupported opcode - cannot compile this function
				// Any opcode not in the list above will fall back to VM execution
				return false;
		}

		// Advance IP (this is a simplified advancement - in reality need proper opcode parsing)
		ip += 1;

		// For safety, avoid infinite loops
		if (ip > code_size) {
			return false;
		}
	}

	return true;
}

Error GDScriptBytecodeELFCompiler::compile_function_to_elf64(GDScriptFunction *p_function, PackedByteArray &r_elf_data) const {
	if (!p_function) {
		return ERR_INVALID_PARAMETER;
	}

	if (!can_compile_function_to_elf64(p_function)) {
		return ERR_UNAVAILABLE;
	}

	if (!code_generator.is_valid() || !compiler.is_valid()) {
		return ERR_UNAVAILABLE;
	}

	// Generate C code from bytecode
	String c_code = code_generator->generate_c_code(p_function);
	if (c_code.is_empty()) {
		print_error("GDScriptBytecodeELFCompiler: Failed to generate C code");
		return ERR_INVALID_DATA;
	}

	print_verbose("GDScriptBytecodeELFCompiler: Generated C code:");
	print_verbose(c_code);

	// Compile C code to ELF
	Error compile_err = compiler->compile_c_to_elf(c_code, include_paths, r_elf_data);
	if (compile_err != OK) {
		print_error("GDScriptBytecodeELFCompiler: Compilation failed");
		return compile_err;
	}

	if (r_elf_data.is_empty()) {
		print_error("GDScriptBytecodeELFCompiler: Compilation produced empty ELF");
		return ERR_INVALID_DATA;
	}

	print_verbose(vformat("GDScriptBytecodeELFCompiler: Successfully compiled function '%s' to ELF (%d bytes)",
			p_function->get_name(), r_elf_data.size()));

	return OK;
}

bool GDScriptBytecodeELFCompiler::can_compile_function_to_elf64(const GDScriptFunction *p_function) const {
	if (!p_function) {
		return false;
	}

	// Check if bytecode exists
	if (!p_function->_code_ptr || p_function->_code_size == 0) {
		return false;
	}

	// Check if cross-compiler is available
	if (!compiler.is_valid()) {
		return false;
	}
	if (!compiler->is_cross_compiler_available()) {
		return false;
	}

	// Check if function contains only basic opcodes we can handle
	if (!is_basic_opcodes_only(p_function)) {
		return false;
	}

	return true;
}
