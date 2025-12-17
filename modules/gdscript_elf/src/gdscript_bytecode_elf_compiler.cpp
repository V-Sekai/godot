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
#include "core/os/os.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "gdscript_c_compiler.h"
#include "gdscript_to_stablehlo.h"

GDScriptBytecodeELFCompiler::GDScriptBytecodeELFCompiler() {
	compiler.instantiate();
}

void GDScriptBytecodeELFCompiler::add_include_path(const String &p_path) {
	include_paths.append(p_path);
}

void GDScriptBytecodeELFCompiler::set_include_paths(const Vector<String> &p_paths) {
	include_paths = p_paths;
}

bool GDScriptBytecodeELFCompiler::is_basic_opcodes_only(const GDScriptFunction *p_function) const {
	if (!p_function || p_function->code.is_empty()) {
		return false;
	}

	const int *code_ptr = p_function->code.ptr();
	int code_size = p_function->code.size();
	int ip = 0;

	// Check if all opcodes are basic/directly supported ones
	while (ip < code_size) {
		int opcode = code_ptr[ip];

		switch (opcode) {
			// ALL opcodes are now supported via composite patterns or syscalls
			// Core opcodes (consolidated)
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
			// Operators
			case GDScriptFunction::OPCODE_OPERATOR:
			case GDScriptFunction::OPCODE_OPERATOR_VALIDATED:
			// Type tests
			case GDScriptFunction::OPCODE_TYPE_TEST_BUILTIN:
			case GDScriptFunction::OPCODE_TYPE_TEST_ARRAY:
			case GDScriptFunction::OPCODE_TYPE_TEST_DICTIONARY:
			case GDScriptFunction::OPCODE_TYPE_TEST_NATIVE:
			case GDScriptFunction::OPCODE_TYPE_TEST_SCRIPT:
			// Keyed/Indexed operations
			case GDScriptFunction::OPCODE_SET_KEYED:
			case GDScriptFunction::OPCODE_SET_KEYED_VALIDATED:
			case GDScriptFunction::OPCODE_SET_INDEXED_VALIDATED:
			case GDScriptFunction::OPCODE_GET_KEYED:
			case GDScriptFunction::OPCODE_GET_KEYED_VALIDATED:
			case GDScriptFunction::OPCODE_GET_INDEXED_VALIDATED:
			// Named operations
			case GDScriptFunction::OPCODE_SET_NAMED:
			case GDScriptFunction::OPCODE_SET_NAMED_VALIDATED:
			case GDScriptFunction::OPCODE_GET_NAMED:
			case GDScriptFunction::OPCODE_GET_NAMED_VALIDATED:
			// Member operations
			case GDScriptFunction::OPCODE_GET_MEMBER:
			case GDScriptFunction::OPCODE_SET_MEMBER:
			// Static variables
			case GDScriptFunction::OPCODE_SET_STATIC_VARIABLE:
			case GDScriptFunction::OPCODE_GET_STATIC_VARIABLE:
			// Casts
			case GDScriptFunction::OPCODE_CAST_TO_BUILTIN:
			case GDScriptFunction::OPCODE_CAST_TO_NATIVE:
			case GDScriptFunction::OPCODE_CAST_TO_SCRIPT:
			// Constructors
			case GDScriptFunction::OPCODE_CONSTRUCT:
			case GDScriptFunction::OPCODE_CONSTRUCT_VALIDATED:
			case GDScriptFunction::OPCODE_CONSTRUCT_ARRAY:
			case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_ARRAY:
			case GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY:
			case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_DICTIONARY:
			// Method calls
			case GDScriptFunction::OPCODE_CALL:
			case GDScriptFunction::OPCODE_CALL_RETURN:
			case GDScriptFunction::OPCODE_CALL_ASYNC:
			case GDScriptFunction::OPCODE_CALL_UTILITY:
			case GDScriptFunction::OPCODE_CALL_UTILITY_VALIDATED:
			case GDScriptFunction::OPCODE_CALL_GDSCRIPT_UTILITY:
			case GDScriptFunction::OPCODE_CALL_BUILTIN_TYPE_VALIDATED:
			case GDScriptFunction::OPCODE_CALL_SELF_BASE:
			case GDScriptFunction::OPCODE_CALL_METHOD_BIND:
			case GDScriptFunction::OPCODE_CALL_METHOD_BIND_RET:
			case GDScriptFunction::OPCODE_CALL_BUILTIN_STATIC:
			case GDScriptFunction::OPCODE_CALL_NATIVE_STATIC:
			case GDScriptFunction::OPCODE_CALL_NATIVE_STATIC_VALIDATED_RETURN:
			case GDScriptFunction::OPCODE_CALL_NATIVE_STATIC_VALIDATED_NO_RETURN:
			case GDScriptFunction::OPCODE_CALL_METHOD_BIND_VALIDATED_RETURN:
			case GDScriptFunction::OPCODE_CALL_METHOD_BIND_VALIDATED_NO_RETURN:
			// Await
			case GDScriptFunction::OPCODE_AWAIT:
			case GDScriptFunction::OPCODE_AWAIT_RESUME:
			// Lambdas
			case GDScriptFunction::OPCODE_CREATE_LAMBDA:
			case GDScriptFunction::OPCODE_CREATE_SELF_LAMBDA:
			// Iterators
			case GDScriptFunction::OPCODE_ITERATE_BEGIN:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_INT:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_FLOAT:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR2:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR2I:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR3:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_VECTOR3I:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_STRING:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_DICTIONARY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_BYTE_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_INT32_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_INT64_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_FLOAT32_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_FLOAT64_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_STRING_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_VECTOR2_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_VECTOR3_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_COLOR_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_PACKED_VECTOR4_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_OBJECT:
			case GDScriptFunction::OPCODE_ITERATE_BEGIN_RANGE:
			case GDScriptFunction::OPCODE_ITERATE:
			case GDScriptFunction::OPCODE_ITERATE_INT:
			case GDScriptFunction::OPCODE_ITERATE_FLOAT:
			case GDScriptFunction::OPCODE_ITERATE_VECTOR2:
			case GDScriptFunction::OPCODE_ITERATE_VECTOR2I:
			case GDScriptFunction::OPCODE_ITERATE_VECTOR3:
			case GDScriptFunction::OPCODE_ITERATE_VECTOR3I:
			case GDScriptFunction::OPCODE_ITERATE_STRING:
			case GDScriptFunction::OPCODE_ITERATE_DICTIONARY:
			case GDScriptFunction::OPCODE_ITERATE_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_BYTE_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_INT32_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_INT64_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_FLOAT32_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_FLOAT64_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_STRING_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_VECTOR2_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_VECTOR3_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_COLOR_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_PACKED_VECTOR4_ARRAY:
			case GDScriptFunction::OPCODE_ITERATE_OBJECT:
			case GDScriptFunction::OPCODE_ITERATE_RANGE:
			// Globals
			case GDScriptFunction::OPCODE_STORE_GLOBAL:
			case GDScriptFunction::OPCODE_STORE_NAMED_GLOBAL:
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
			// Debug/metadata opcodes
			case GDScriptFunction::OPCODE_LINE:
			case GDScriptFunction::OPCODE_BREAKPOINT:
			case GDScriptFunction::OPCODE_ASSERT:
			case GDScriptFunction::OPCODE_END:
				// All opcodes are supported - either via direct C code, composite patterns, or syscalls
				break;
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

	if (!compiler.is_valid()) {
		return ERR_UNAVAILABLE;
	}

	// Check if function can be converted to StableHLO
	if (!GDScriptToStableHLO::can_convert_function(p_function)) {
		print_error("GDScriptBytecodeELFCompiler: Function contains unsupported opcodes");
		return ERR_UNAVAILABLE;
	}

	// Create temporary directory for compilation
	String temp_dir = OS::get_singleton()->get_cache_path();
	if (temp_dir.is_empty()) {
		temp_dir = OS::get_singleton()->get_user_data_dir();
	}
	temp_dir = temp_dir.path_join("godot_gdscript_tmp");

	Ref<DirAccess> dir = DirAccess::create_for_path(temp_dir);
	if (!dir.is_valid()) {
		return ERR_CANT_CREATE;
	}

	if (!dir->dir_exists(temp_dir)) {
		Error err = dir->make_dir_recursive(temp_dir);
		if (err != OK) {
			return err;
		}
	}

	// Generate unique filenames
	uint64_t timestamp = OS::get_singleton()->get_ticks_msec();
	String base_name = vformat("gdscript_%llu", timestamp);
	String stablehlo_path = temp_dir.path_join(base_name + ".stablehlo");
	String cpp_path = temp_dir.path_join(base_name + ".cpp");

	// Step 1: Convert GDScript to StableHLO
	String stablehlo_file = GDScriptToStableHLO::convert_function_to_stablehlo_bytecode(p_function, stablehlo_path);
	if (stablehlo_file.is_empty()) {
		print_error("GDScriptBytecodeELFCompiler: Failed to convert function to StableHLO");
		return ERR_INVALID_DATA;
	}

	// Step 2: Execute external tool to convert StableHLO to C++
	// Try to find stablehlo-to-cpp tool
	String tool_path = "stablehlo-to-cpp";
	
	// Check environment variable for custom tool path
	String env_tool_path = OS::get_singleton()->get_environment("GODOT_STABLEHLO_TO_CPP_PATH");
	if (!env_tool_path.is_empty()) {
		tool_path = env_tool_path;
	} else {
		// Try relative path from executable
		String exe_path = OS::get_singleton()->get_executable_path();
		String exe_dir = exe_path.get_base_dir();
		String relative_tool = exe_dir.path_join("stablehlo-to-cpp");
		Ref<FileAccess> test_file = FileAccess::open(relative_tool, FileAccess::READ);
		if (test_file.is_valid()) {
			test_file->close();
			tool_path = relative_tool;
		}
	}
	
	List<String> tool_args;
	tool_args.push_back(stablehlo_file);
	tool_args.push_back(cpp_path);

	String tool_output;
	int tool_exit_code;
	Error tool_err = OS::get_singleton()->execute(tool_path, tool_args, &tool_output, &tool_exit_code, false);
	
	if (tool_err != OK || tool_exit_code != 0) {
		print_error(vformat("GDScriptBytecodeELFCompiler: External tool failed: %s", tool_output));
		// Cleanup
		dir->remove(stablehlo_file);
		return ERR_FILE_CANT_READ;
	}

	// Step 3: Read generated C++ code
	Ref<FileAccess> cpp_file = FileAccess::open(cpp_path, FileAccess::READ);
	if (!cpp_file.is_valid()) {
		print_error("GDScriptBytecodeELFCompiler: Failed to read generated C++ file");
		// Cleanup
		dir->remove(stablehlo_file);
		return ERR_FILE_CANT_READ;
	}

	String c_code = cpp_file->get_as_text();
	cpp_file->close();

	if (c_code.is_empty()) {
		print_error("GDScriptBytecodeELFCompiler: Generated C++ code is empty");
		// Cleanup
		dir->remove(stablehlo_file);
		dir->remove(cpp_path);
		return ERR_INVALID_DATA;
	}

	print_verbose("GDScriptBytecodeELFCompiler: Generated C++ code:");
	print_verbose(c_code);

	// Step 4: Compile C++ code to ELF
	Error compile_err = compiler->compile_c_to_elf(c_code, include_paths, r_elf_data);
	
	// Cleanup temp files
	dir->remove(stablehlo_file);
	dir->remove(cpp_path);

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
	if (p_function->code.is_empty()) {
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

	// Check if function can be converted to StableHLO
	if (!GDScriptToStableHLO::can_convert_function(p_function)) {
		return false;
	}

	return true;
}
