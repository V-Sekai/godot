/**************************************************************************/
/*  gdscript_to_c99.cpp                                                   */
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
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gdscript_to_c99.h"

#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/variant/array.h"
#include "gdscript_function.h"
#include "gdscript.h"

String GDScriptToC99::gdscript_type_to_c99(const GDScriptDataType &p_type) {
	if (!p_type.has_type()) {
		return "void*"; // Variant - use void* for now
	}
	
	switch (p_type.kind) {
		case GDScriptDataType::BUILTIN: {
			switch (p_type.builtin_type) {
				case Variant::BOOL:
					return "bool";
				case Variant::INT:
					return "int64_t";
				case Variant::FLOAT:
					return "double";
				case Variant::STRING:
					return "char*";
				case Variant::ARRAY:
				case Variant::DICTIONARY:
				case Variant::OBJECT:
				default:
					return "void*";
			}
		}
		case GDScriptDataType::NATIVE:
		case GDScriptDataType::SCRIPT:
		case GDScriptDataType::GDSCRIPT:
		case GDScriptDataType::VARIANT:
		default:
			return "void*";
	}
}

String GDScriptToC99::generate_syscall_wrappers() {
	return R"(
// Godot syscall wrappers for RISC-V using inline assembly
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Syscall numbers
#define GODOT_SYSCALL_PRINT 1
#define GODOT_SYSCALL_VCALL 2
#define GODOT_SYSCALL_VCREATE 3
#define GODOT_SYSCALL_VASSIGN 4
#define GODOT_SYSCALL_VASSIGN_KEYED 5
#define GODOT_SYSCALL_VASSIGN_INDEXED 6
#define GODOT_SYSCALL_OBJ_PROP_GET 7
#define GODOT_SYSCALL_OBJ_PROP_SET 8
#define GODOT_SYSCALL_TYPE_TEST 9

// Print string (byte buffer) - uses inline assembly for syscall
static inline void godot_syscall_print(const char* str, size_t len) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_PRINT;
    register int64_t a0 __asm__("a0") = (int64_t)str;
    register int64_t a1 __asm__("a1") = (int64_t)len;
    __asm__ volatile (
        "ecall"
        :
        : "r" (a7), "r" (a0), "r" (a1)
        : "memory"
    );
}

// Virtual call - uses inline assembly for syscall
static inline void* godot_vcall(void* obj, int64_t method_id, void** args, int64_t arg_count) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_VCALL;
    register int64_t a0 __asm__("a0") = (int64_t)obj;
    register int64_t a1 __asm__("a1") = method_id;
    register int64_t a2 __asm__("a2") = (int64_t)args;
    register int64_t a3 __asm__("a3") = arg_count;
    register int64_t result __asm__("a0");
    __asm__ volatile (
        "ecall"
        : "=r" (result)
        : "r" (a7), "r" (a0), "r" (a1), "r" (a2), "r" (a3)
        : "memory"
    );
    return (void*)result;
}

// Variant create - uses inline assembly for syscall
static inline void* godot_vcreate(int64_t type) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_VCREATE;
    register int64_t a0 __asm__("a0") = type;
    register int64_t result __asm__("a0");
    __asm__ volatile (
        "ecall"
        : "=r" (result)
        : "r" (a7), "r" (a0)
        : "memory"
    );
    return (void*)result;
}

// Variant assign - uses inline assembly for syscall
static inline void godot_vassign(void* dst, void* src) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_VASSIGN;
    register int64_t a0 __asm__("a0") = (int64_t)dst;
    register int64_t a1 __asm__("a1") = (int64_t)src;
    __asm__ volatile (
        "ecall"
        :
        : "r" (a7), "r" (a0), "r" (a1)
        : "memory"
    );
}

// Variant assign keyed - uses inline assembly for syscall
static inline void godot_vassign_keyed(void* dst, void* key, void* value) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_VASSIGN_KEYED;
    register int64_t a0 __asm__("a0") = (int64_t)dst;
    register int64_t a1 __asm__("a1") = (int64_t)key;
    register int64_t a2 __asm__("a2") = (int64_t)value;
    __asm__ volatile (
        "ecall"
        :
        : "r" (a7), "r" (a0), "r" (a1), "r" (a2)
        : "memory"
    );
}

// Variant assign indexed - uses inline assembly for syscall
static inline void godot_vassign_indexed(void* dst, int64_t index, void* value) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_VASSIGN_INDEXED;
    register int64_t a0 __asm__("a0") = (int64_t)dst;
    register int64_t a1 __asm__("a1") = index;
    register int64_t a2 __asm__("a2") = (int64_t)value;
    __asm__ volatile (
        "ecall"
        :
        : "r" (a7), "r" (a0), "r" (a1), "r" (a2)
        : "memory"
    );
}

// Object property get - uses inline assembly for syscall
static inline void* godot_obj_prop_get(void* obj, int64_t prop_id) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_OBJ_PROP_GET;
    register int64_t a0 __asm__("a0") = (int64_t)obj;
    register int64_t a1 __asm__("a1") = prop_id;
    register int64_t result __asm__("a0");
    __asm__ volatile (
        "ecall"
        : "=r" (result)
        : "r" (a7), "r" (a0), "r" (a1)
        : "memory"
    );
    return (void*)result;
}

// Object property set - uses inline assembly for syscall
static inline void godot_obj_prop_set(void* obj, int64_t prop_id, void* value) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_OBJ_PROP_SET;
    register int64_t a0 __asm__("a0") = (int64_t)obj;
    register int64_t a1 __asm__("a1") = prop_id;
    register int64_t a2 __asm__("a2") = (int64_t)value;
    __asm__ volatile (
        "ecall"
        :
        : "r" (a7), "r" (a0), "r" (a1), "r" (a2)
        : "memory"
    );
}

// Type test - uses inline assembly for syscall
static inline bool godot_type_test(void* value, int64_t type) {
    register int64_t a7 __asm__("a7") = GODOT_SYSCALL_TYPE_TEST;
    register int64_t a0 __asm__("a0") = (int64_t)value;
    register int64_t a1 __asm__("a1") = type;
    register int64_t result __asm__("a0");
    __asm__ volatile (
        "ecall"
        : "=r" (result)
        : "r" (a7), "r" (a0), "r" (a1)
        : "memory"
    );
    return (bool)result;
}

)";
}

String GDScriptToC99::generate_function_signature(const GDScriptFunction *p_func) {
	String sig;
	
	// Return type
	String c99_return = gdscript_type_to_c99(p_func->return_type);
	sig += c99_return + " ";
	
	// Function name
	String func_name = p_func->get_name().operator String();
	if (func_name.is_empty()) {
		func_name = "gdscript_function";
	}
	// Sanitize function name for C
	func_name = func_name.replace("@", "_at_").replace(".", "_");
	sig += func_name + "(";
	
	// Arguments
	for (int i = 0; i < p_func->argument_types.size(); i++) {
		if (i > 0) {
			sig += ", ";
		}
		String param_type = gdscript_type_to_c99(p_func->argument_types[i]);
		sig += param_type + " arg" + String::num(i);
	}
	
	sig += ")";
	return sig;
}

String GDScriptToC99::generate_operation_c99(const GDScriptFunction *p_func, int p_ip, HashMap<int, String> &p_stack_vars, int &p_next_var) {
	if (!p_func || p_func->code.is_empty()) {
		return String();
	}
	
	const int *code_ptr = p_func->code.ptr();
	int code_size = p_func->code.size();
	
	if (p_ip >= code_size) {
		return String();
	}
	
	int opcode = code_ptr[p_ip];
	String result;
	
	// Helper to get constant value
	auto get_constant = [&](int idx) -> Variant {
		if (idx >= 0 && idx < p_func->constants.size()) {
			return p_func->constants[idx];
		}
		return Variant();
	};
	
	// Helper to decode address
	auto decode_address = [](int addr) -> Pair<int, int> {
		int type = (addr >> GDScriptFunction::ADDR_BITS) & 0xFF;
		int index = addr & GDScriptFunction::ADDR_MASK;
		return Pair<int, int>(type, index);
	};
	
	// Helper to get stack variable name
	auto get_stack_var = [&](int addr) -> String {
		if (p_stack_vars.has(addr)) {
			return p_stack_vars[addr];
		}
		Pair<int, int> decoded = decode_address(addr);
		int type = decoded.first;
		int index = decoded.second;
		
		if (type == (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS)) {
			// Constant
			Variant const_val = get_constant(index);
			if (const_val.get_type() == Variant::FLOAT) {
				return String::num(const_val);
			} else if (const_val.get_type() == Variant::INT) {
				return String::num(const_val);
			} else if (const_val.get_type() == Variant::STRING) {
				return "\"" + String(const_val) + "\"";
			}
		}
		
		String var_name = "var" + String::num(p_next_var++);
		p_stack_vars[addr] = var_name;
		return var_name;
	};
	
	switch (opcode) {
		case GDScriptFunction::OPCODE_RETURN: {
			if (p_ip + 1 < code_size) {
				int return_count = code_ptr[p_ip + 1];
				if (return_count > 0 && p_ip + 2 < code_size) {
					int return_addr = code_ptr[p_ip + 2];
					String return_val = get_stack_var(return_addr);
					result = "return " + return_val + ";\n";
				} else {
					result = "return;\n";
				}
			} else {
				result = "return;\n";
			}
		} break;
		
		case GDScriptFunction::OPCODE_ASSIGN: {
			if (p_ip + 3 < code_size) {
				int dst = code_ptr[p_ip + 1];
				int src = code_ptr[p_ip + 2];
				String dst_var = get_stack_var(dst);
				String src_var = get_stack_var(src);
				result = dst_var + " = " + src_var + ";\n";
			}
		} break;
		
		case GDScriptFunction::OPCODE_OPERATOR: {
			if (p_ip + 5 < code_size) {
				int dst = code_ptr[p_ip + 1];
				int left = code_ptr[p_ip + 2];
				int right = code_ptr[p_ip + 3];
				Variant::Operator op = (Variant::Operator)code_ptr[p_ip + 4];
				
				String dst_var = get_stack_var(dst);
				String left_var = get_stack_var(left);
				String right_var = get_stack_var(right);
				
				String op_str;
				switch (op) {
					case Variant::OP_ADD:
						op_str = "+";
						break;
					case Variant::OP_SUBTRACT:
						op_str = "-";
						break;
					case Variant::OP_MULTIPLY:
						op_str = "*";
						break;
					case Variant::OP_DIVIDE:
						op_str = "/";
						break;
					default:
						op_str = "/* unknown op */";
						break;
				}
				
				result = dst_var + " = " + left_var + " " + op_str + " " + right_var + ";\n";
			}
		} break;
		
		case GDScriptFunction::OPCODE_CALL: {
			if (p_ip + 2 < code_size) {
				int arg_count = code_ptr[p_ip + 1];
				// For now, placeholder - would need to handle method calls
				result = "/* call operation - TODO */\n";
			}
		} break;
		
		default:
			result = "/* opcode " + String::num(opcode) + " - TODO */\n";
			break;
	}
	
	return result;
}

String GDScriptToC99::generate_function_body(const GDScriptFunction *p_func, HashMap<int, String> &p_stack_vars) {
	String body = "{\n";
	
	if (!p_func || p_func->code.is_empty()) {
		body += "    return;\n";
		body += "}\n";
		return body;
	}
	
	const int *code_ptr = p_func->code.ptr();
	int code_size = p_func->code.size();
	int next_var = 0;
	
	// Declare stack variables
	body += "    // Stack variables\n";
	for (int i = 0; i < code_size; i++) {
		int opcode = code_ptr[i];
		// Extract stack addresses from operations
		// This is simplified - would need proper analysis
	}
	
	// Helper to get opcode size (copied from StableHLO converter)
	auto get_opcode_size = [](int opcode, const int *code_ptr, int ip, int code_size) -> int {
		switch (opcode) {
			case GDScriptFunction::OPCODE_ASSIGN_NULL:
			case GDScriptFunction::OPCODE_ASSIGN_TRUE:
			case GDScriptFunction::OPCODE_ASSIGN_FALSE:
			case GDScriptFunction::OPCODE_LINE:
			case GDScriptFunction::OPCODE_BREAKPOINT:
			case GDScriptFunction::OPCODE_ASSERT:
			case GDScriptFunction::OPCODE_END:
				return 1;
			case GDScriptFunction::OPCODE_JUMP:
			case GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT:
				return 2;
			case GDScriptFunction::OPCODE_JUMP_IF:
			case GDScriptFunction::OPCODE_JUMP_IF_NOT:
			case GDScriptFunction::OPCODE_JUMP_IF_SHARED:
			case GDScriptFunction::OPCODE_ASSIGN:
			case GDScriptFunction::OPCODE_GET_MEMBER:
			case GDScriptFunction::OPCODE_SET_MEMBER:
				return 3;
			case GDScriptFunction::OPCODE_RETURN:
				if (ip + 1 < code_size) {
					int return_count = code_ptr[ip + 1];
					return 2 + (return_count > 0 ? 1 : 0);
				}
				return 2;
			case GDScriptFunction::OPCODE_OPERATOR_VALIDATED:
				return 5;
			case GDScriptFunction::OPCODE_OPERATOR: {
				constexpr int ptr_size = sizeof(Variant::ValidatedOperatorEvaluator) / sizeof(int);
				return 5 + ptr_size;
			}
			case GDScriptFunction::OPCODE_CALL_RETURN:
			case GDScriptFunction::OPCODE_CALL: {
				if (ip + 1 < code_size) {
					int arg_count = code_ptr[ip + 1];
					return 2 + arg_count;
				}
				return 2;
			}
			default:
				return 1; // Default: skip one word
		}
	};
	
	// Generate code for each operation
	body += "    // Function body\n";
	for (int ip = 0; ip < code_size; ) {
		String op_code = generate_operation_c99(p_func, ip, p_stack_vars, next_var);
		if (!op_code.is_empty()) {
			body += "    " + op_code;
		}
		
		// Advance IP by opcode size
		int opcode = code_ptr[ip];
		int opcode_size = get_opcode_size(opcode, code_ptr, ip, code_size);
		ip += opcode_size;
	}
	
	body += "}\n";
	return body;
}

bool GDScriptToC99::can_convert_to_c99(const GDScriptFunction *p_func) {
	if (!p_func) {
		return false;
	}
	
	// Check if function has bytecode
	if (p_func->code.is_empty()) {
		return false;
	}
	
	// For now, allow all functions (can add restrictions later)
	return true;
}

String GDScriptToC99::generate_c99(const GDScriptFunction *p_func) {
	if (!can_convert_to_c99(p_func)) {
		return String();
	}
	
	String c99_code;
	
	// Header includes
	c99_code += "#include <stdint.h>\n";
	c99_code += "#include <stdbool.h>\n";
	c99_code += "#include <stddef.h>\n";
	c99_code += "\n";
	
	// Syscall wrappers
	c99_code += generate_syscall_wrappers();
	c99_code += "\n";
	
	// Function signature
	String func_sig = generate_function_signature(p_func);
	c99_code += func_sig;
	c99_code += "\n";
	
	// Function body
	HashMap<int, String> stack_vars;
	String func_body = generate_function_body(p_func, stack_vars);
	c99_code += func_body;
	
	return c99_code;
}

PackedByteArray GDScriptToC99::compile_c99_to_elf64(const String &p_c99_code, const String &p_function_name, bool p_debug) {
	// Create temporary directory
	String temp_dir = OS::get_singleton()->get_cache_path().path_join("gdscript_c99_compile");
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (dir.is_valid()) {
		dir->make_dir_recursive(temp_dir);
	}
	
	String c_file = temp_dir.path_join(p_function_name + ".c");
	String elf_file = temp_dir.path_join(p_function_name + ".elf");
	
	// Write C99 code to file
	Ref<FileAccess> c_f = FileAccess::open(c_file, FileAccess::WRITE);
	if (!c_f.is_valid()) {
		print_error("GDScriptToC99: Failed to create temp C file");
		return PackedByteArray();
	}
	c_f->store_string(p_c99_code);
	c_f->close();
	
	// Try to compile with external RISC-V GCC/Clang
	String compiler = OS::get_singleton()->get_environment("RISCV_CC");
	if (compiler.is_empty()) {
		compiler = "riscv64-unknown-elf-gcc";
	}
	
	// Compile command
	Vector<String> args;
	args.push_back("-std=c99");
	args.push_back("-O2");
	args.push_back("-nostdlib");
	args.push_back("-ffreestanding");
	if (p_debug) {
		args.push_back("-g");
	}
	args.push_back("-o");
	args.push_back(elf_file);
	args.push_back(c_file);
	
	List<String> args_list;
	for (int i = 0; i < args.size(); i++) {
		args_list.push_back(args[i]);
	}
	
	String output;
	int exit_code;
	Error error = OS::get_singleton()->execute(compiler, args_list, &output, &exit_code, true);
	
	if (error != OK || exit_code != 0) {
		print_error("GDScriptToC99: Failed to compile C99 to ELF64");
		print_error("Compiler: " + compiler);
		if (!output.is_empty()) {
			print_error("Output: " + output);
		}
		// TODO: Try TCC if available
		return PackedByteArray();
	}
	
	// Read ELF binary
	Ref<FileAccess> elf_f = FileAccess::open(elf_file, FileAccess::READ);
	if (!elf_f.is_valid()) {
		print_error("GDScriptToC99: Failed to read ELF file");
		return PackedByteArray();
	}
	
	PackedByteArray elf_binary;
	elf_binary.resize(elf_f->get_length());
	elf_f->get_buffer(elf_binary.ptrw(), elf_binary.size());
	elf_f->close();
	
	return elf_binary;
}

void GDScriptToC99::_bind_methods() {
	// Instance methods
	ClassDB::bind_method(D_METHOD("convert_script_function_to_c99", "script", "function_name"), &GDScriptToC99::convert_script_function_to_c99);
	ClassDB::bind_method(D_METHOD("can_convert_script_function", "script", "function_name"), &GDScriptToC99::can_convert_script_function);
	
	// Static methods - exposed for direct access from GDScript
	ClassDB::bind_static_method("GDScriptToC99", D_METHOD("generate_c99_from_script", "script", "function_name"), &GDScriptToC99::generate_c99_from_script);
	ClassDB::bind_static_method("GDScriptToC99", D_METHOD("can_convert_script", "script", "function_name"), &GDScriptToC99::can_convert_script);
	ClassDB::bind_static_method("GDScriptToC99", D_METHOD("compile_c99_to_elf64", "c99_code", "function_name", "debug"), &GDScriptToC99::compile_c99_to_elf64, DEFVAL(true));
}

String GDScriptToC99::convert_script_function_to_c99(Ref<GDScript> p_script, const StringName &p_function_name) const {
	if (!p_script.is_valid()) {
		print_error("GDScriptToC99: Invalid script");
		return String();
	}
	
	const HashMap<StringName, GDScriptFunction *> &funcs = p_script->get_member_functions();
	if (!funcs.has(p_function_name)) {
		print_error(vformat("GDScriptToC99: Function '%s' not found in script", p_function_name));
		return String();
	}
	
	GDScriptFunction *func = funcs.get(p_function_name);
	if (func == nullptr) {
		print_error(vformat("GDScriptToC99: Function '%s' is null", p_function_name));
		return String();
	}
	
	return generate_c99(func);
}

bool GDScriptToC99::can_convert_script_function(Ref<GDScript> p_script, const StringName &p_function_name) const {
	return can_convert_script(p_script, p_function_name);
}

String GDScriptToC99::generate_c99_from_script(Ref<GDScript> p_script, const StringName &p_function_name) {
	if (!p_script.is_valid()) {
		print_error("GDScriptToC99: Invalid script");
		return String();
	}
	
	const HashMap<StringName, GDScriptFunction *> &funcs = p_script->get_member_functions();
	if (!funcs.has(p_function_name)) {
		print_error(vformat("GDScriptToC99: Function '%s' not found in script", p_function_name));
		return String();
	}
	
	GDScriptFunction *func = funcs.get(p_function_name);
	if (func == nullptr) {
		print_error(vformat("GDScriptToC99: Function '%s' is null", p_function_name));
		return String();
	}
	
	return generate_c99(func);
}

bool GDScriptToC99::can_convert_script(Ref<GDScript> p_script, const StringName &p_function_name) {
	if (!p_script.is_valid()) {
		return false;
	}
	
	const HashMap<StringName, GDScriptFunction *> &funcs = p_script->get_member_functions();
	if (!funcs.has(p_function_name)) {
		return false;
	}
	
	GDScriptFunction *func = funcs.get(p_function_name);
	if (func == nullptr) {
		return false;
	}
	
	return can_convert_to_c99(func);
}
