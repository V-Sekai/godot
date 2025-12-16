/**************************************************************************/
/*  gdscript_bytecode_serializer.cpp                                      */
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

#include "gdscript_bytecode_serializer.h"
#include "core/io/file_access.h"

bool GDScriptBytecodeSerializer::is_basic_opcode(int p_opcode) {
	// ~14 core opcodes for simplified execution
	switch (p_opcode) {
		// Returns (1)
		case GDScriptFunction::OPCODE_RETURN:
		// Assignments (1) - all ASSIGN variants consolidated
		case GDScriptFunction::OPCODE_ASSIGN:
		case GDScriptFunction::OPCODE_ASSIGN_NULL:
		case GDScriptFunction::OPCODE_ASSIGN_TRUE:
		case GDScriptFunction::OPCODE_ASSIGN_FALSE:
		// Control Flow (3)
		case GDScriptFunction::OPCODE_JUMP:
		case GDScriptFunction::OPCODE_JUMP_IF:
		case GDScriptFunction::OPCODE_JUMP_IF_NOT:
		// Operators (2)
		case GDScriptFunction::OPCODE_OPERATOR:
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED:
		// Property Access (2)
		case GDScriptFunction::OPCODE_GET_MEMBER:
		case GDScriptFunction::OPCODE_SET_MEMBER:
		// Method Calls (1) - basic CALL
		case GDScriptFunction::OPCODE_CALL:
		case GDScriptFunction::OPCODE_CALL_RETURN:
		// Debug/Metadata (4)
		case GDScriptFunction::OPCODE_LINE:
		case GDScriptFunction::OPCODE_BREAKPOINT:
		case GDScriptFunction::OPCODE_ASSERT:
		case GDScriptFunction::OPCODE_END:
			return true;
		default:
			return false;
	}
}

bool GDScriptBytecodeSerializer::uses_basic_opcodes_only(const GDScriptFunction *p_function) {
	if (!p_function || !p_function->_code_ptr || p_function->_code_size == 0) {
		return false;
	}

	const int *code_ptr = p_function->_code_ptr;
	int code_size = p_function->_code_size;
	int ip = 0;

	while (ip < code_size) {
		int opcode = code_ptr[ip];
		if (!is_basic_opcode(opcode)) {
			return false;
		}
		
		// Advance IP (simplified - would need proper opcode parsing for exact advancement)
		ip += 1;
		if (ip > code_size) {
			return false;
		}
	}

	return true;
}

PackedByteArray GDScriptBytecodeSerializer::serialize_function(const GDScriptFunction *p_function) {
	if (!p_function || !p_function->_code_ptr || p_function->_code_size == 0) {
		return PackedByteArray();
	}

	// Simple binary format:
	// - Header: function_name (String), stack_size (int), argument_count (int), is_static (bool), code_size (int)
	// - Opcodes: code_size * sizeof(int)
	// - Constants: count (int) + Variant array
	// - Global names: count (int) + StringName array

	PackedByteArray data;
	Ref<FileAccess> memfile = FileAccess::open_internal("user://temp_serialize", FileAccess::WRITE);
	if (!memfile.is_valid()) {
		return PackedByteArray();
	}

	// Write header
	String function_name = p_function->get_name();
	memfile->store_pascal_string(function_name);
	memfile->store_32(p_function->get_max_stack_size());
	memfile->store_32(p_function->get_argument_count());
	memfile->store_8(p_function->is_static() ? 1 : 0);
	memfile->store_32(p_function->_code_size);

	// Write opcodes
	for (int i = 0; i < p_function->_code_size; i++) {
		memfile->store_32(p_function->_code_ptr[i]);
	}

	// Write constants
	memfile->store_32(p_function->constants.size());
	for (int i = 0; i < p_function->constants.size(); i++) {
		memfile->store_var(p_function->constants[i]);
	}

	// Write global names
	memfile->store_32(p_function->global_names.size());
	for (int i = 0; i < p_function->global_names.size(); i++) {
		memfile->store_pascal_string(p_function->global_names[i]);
	}

	memfile->close();
	
	// Read back as PackedByteArray
	Ref<FileAccess> readfile = FileAccess::open_internal("user://temp_serialize", FileAccess::READ);
	if (!readfile.is_valid()) {
		return PackedByteArray();
	}
	
	int64_t file_size = readfile->get_length();
	data.resize(file_size);
	readfile->get_buffer(data.ptrw(), file_size);
	readfile->close();
	
	// Cleanup temp file
	FileAccess::remove("user://temp_serialize");

	return data;
}

GDScriptBytecodeSerializer::DeserializedFunction GDScriptBytecodeSerializer::deserialize_function(const PackedByteArray &p_data) {
	DeserializedFunction result;

	if (p_data.is_empty()) {
		return result;
	}

	Ref<FileAccess> memfile = FileAccess::open_internal("user://temp_deserialize", FileAccess::WRITE);
	if (!memfile.is_valid()) {
		return result;
	}
	
	memfile->store_buffer(p_data.ptr(), p_data.size());
	memfile->close();

	Ref<FileAccess> readfile = FileAccess::open_internal("user://temp_deserialize", FileAccess::READ);
	if (!readfile.is_valid()) {
		return result;
	}

	// Read header
	result.function_name = readfile->get_pascal_string();
	result.stack_size = readfile->get_32();
	result.argument_count = readfile->get_32();
	result.is_static = readfile->get_8() != 0;
	int code_size = readfile->get_32();

	// Read opcodes
	result.opcodes.resize(code_size);
	for (int i = 0; i < code_size; i++) {
		result.opcodes.write[i] = readfile->get_32();
	}

	// Read constants
	int constant_count = readfile->get_32();
	result.constants.resize(constant_count);
	for (int i = 0; i < constant_count; i++) {
		result.constants.write[i] = readfile->get_var();
	}

	// Read global names
	int global_name_count = readfile->get_32();
	result.global_names.resize(global_name_count);
	for (int i = 0; i < global_name_count; i++) {
		result.global_names.write[i] = readfile->get_pascal_string();
	}

	readfile->close();
	
	// Cleanup temp file
	FileAccess::remove("user://temp_deserialize");

	return result;
}

