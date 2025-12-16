/**************************************************************************/
/*  gdscript_bytecode_c_code_generator.cpp                                */
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

#include "gdscript_bytecode_c_code_generator.h"
#include "core/error/error_macros.h"
#include "modules/sandbox/src/syscalls.h"

#ifdef MODULE_SANDBOX_ENABLED
#include "modules/sandbox/src/syscalls.h"
#endif

String GDScriptBytecodeCCodeGenerator::generate_function_signature(const String &p_function_name, bool p_is_static) const {
	// Generate C function signature that matches sandbox expectations
	return vformat("void gdscript_%s(void* instance, Variant* args, int argcount, Variant* result, Variant* constants, Variant::ValidatedOperatorEvaluator* operator_funcs)", p_function_name);
}

String GDScriptBytecodeCCodeGenerator::generate_prelogue(int p_stack_size, const GDScriptFunction *p_function) const {
	String prologue = "{\n";
	prologue += vformat("    Variant stack[%d];\n", p_stack_size);
	prologue += "    int ip = 0;\n";
	prologue += "\n";

	// Initialize stack with null variants
	prologue += "    // Initialize stack\n";
	for (int i = 0; i < p_stack_size; i++) {
		prologue += vformat("    stack[%d] = Variant();\n", i);
	}
	prologue += "\n";

	return prologue;
}

String GDScriptBytecodeCCodeGenerator::generate_epilogue() const {
	String epilogue = "}\n";
	return epilogue;
}

String GDScriptBytecodeCCodeGenerator::resolve_address(int p_address, const GDScriptFunction *p_function, bool p_is_destination) const {
	// Convert bytecode address to C variable access
	int addr_type = p_address & GDScriptFunction::ADDR_MASK;
	int addr_index = p_address >> GDScriptFunction::ADDR_BITS;

	switch (addr_type) {
		case GDScriptFunction::ADDR_TYPE_STACK: {
			return vformat("stack[%d]", addr_index);
		}
		case GDScriptFunction::ADDR_TYPE_CONSTANT: {
			return vformat("constants[%d]", addr_index);
		}
		case GDScriptFunction::ADDR_TYPE_MEMBER: {
			// For now, members are handled via syscalls only
			// This would need global name resolution
			String member_name = p_function->get_global_name(addr_index);
			return vformat("get_global_name_cstr(%d)", addr_index); // Placeholder for syscall
		}
		default: {
			return "Variant()"; // Error case
		}
	}
}

String GDScriptBytecodeCCodeGenerator::generate_syscall(int p_ecall_number, const Vector<String> &p_args) const {
	String syscall_code;
	syscall_code += "// Syscall " + itos(p_ecall_number) + "\n";

	// Set up registers for syscall (inline assembly style for C code)
	for (int i = 0; i < p_args.size() && i < 5; i++) {
		syscall_code += vformat("register Variant* a%d asm(\"a%d\") = &(%s);\n", i, i, p_args[i]);
	}

	syscall_code += vformat("register int syscall_number asm(\"a7\") = %d;\n", p_ecall_number);
	syscall_code += "__asm__ volatile(\"ecall\" : : \"r\"(syscall_number)";

	for (int i = 0; i < p_args.size() && i < 5; i++) {
		syscall_code += vformat(", \"r\"(a%d)", i);
	}
	syscall_code += ");\n\n";

	return syscall_code;
}

void GDScriptBytecodeCCodeGenerator::generate_jump_labels(const int *p_code_ptr, int p_code_size, HashMap<int, int> &r_jump_labels) const {
	int label_count = 0;

	// Scan for jump targets to assign labels
	for (int ip = 0; ip < p_code_size;) {
		int opcode = p_code_ptr[ip];
		switch (opcode) {
			case GDScriptFunction::OPCODE_JUMP: {
				if (ip + 1 < p_code_size) {
					int target_ip = p_code_ptr[ip + 1];
					if (!r_jump_labels.has(target_ip)) {
						r_jump_labels[target_ip] = label_count++;
					}
				}
				ip += 2;
				break;
			}
			case GDScriptFunction::OPCODE_JUMP_IF:
			case GDScriptFunction::OPCODE_JUMP_IF_NOT: {
				if (ip + 1 < p_code_size) {
					int target_ip = p_code_ptr[ip + 1];
					if (!r_jump_labels.has(target_ip)) {
						r_jump_labels[target_ip] = label_count++;
					}
				}
				ip += 1; // These opcodes take 1 argument
				break;
			}
			default: {
				// Most opcodes advance by 1
				ip += 1;
				break;
			}
		}
	}
}

String GDScriptBytecodeCCodeGenerator::generate_opcode(GDScriptFunction::Opcode p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size, const GDScriptFunction *p_function) const {
	String opcode_code;

	switch (p_opcode) {
		case GDScriptFunction::OPCODE_RETURN: {
			if (p_ip + 1 < p_code_size) {
				int return_addr = p_code_ptr[p_ip + 1];
				String return_expr = resolve_address(return_addr, p_function);
				opcode_code += vformat("    *result = %s;\n", return_expr.utf8().get_data());
			}
			opcode_code += "    return;\n";
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN:
		case GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY:
		case GDScriptFunction::OPCODE_RETURN_TYPED_DICTIONARY:
		case GDScriptFunction::OPCODE_RETURN_TYPED_NATIVE:
		case GDScriptFunction::OPCODE_RETURN_TYPED_SCRIPT: {
			// Typed returns - for now use regular return (type checking happens in VM)
			p_ip += 2; // value + type info
			if (p_ip - 1 < p_code_size) {
				int return_addr = p_code_ptr[p_ip - 1];
				String return_expr = resolve_address(return_addr, p_function);
				opcode_code += vformat("    *result = %s;\n", return_expr.utf8().get_data());
			}
			opcode_code += "    return;\n";
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN: {
			if (p_ip + 2 < p_code_size) {
				int dst_addr = p_code_ptr[p_ip + 1];
				int src_addr = p_code_ptr[p_ip + 2];
				String dst = resolve_address(dst_addr, p_function, true);
				String src = resolve_address(src_addr, p_function);
				opcode_code += vformat("    %s = %s;\n", dst.utf8().get_data(), src.utf8().get_data());
			}
			p_ip += 3; // opcode + src + dst
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN_NULL: {
			if (p_ip + 1 < p_code_size) {
				int dst_addr = p_code_ptr[p_ip + 1];
				String dst = resolve_address(dst_addr, p_function, true);
				opcode_code += vformat("    %s = Variant();\n", dst.utf8().get_data());
			}
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN_TRUE: {
			if (p_ip + 1 < p_code_size) {
				int dst_addr = p_code_ptr[p_ip + 1];
				String dst = resolve_address(dst_addr, p_function, true);
				opcode_code += vformat("    %s = true;\n", dst.utf8().get_data());
			}
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN_FALSE: {
			if (p_ip + 1 < p_code_size) {
				int dst_addr = p_code_ptr[p_ip + 1];
				String dst = resolve_address(dst_addr, p_function, true);
				opcode_code += vformat("    %s = false;\n", dst.utf8().get_data());
			}
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_ARRAY:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_DICTIONARY:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT: {
			// Typed assignments - for now use regular assignment (type checking happens in bytecode generation)
			if (p_ip + 2 < p_code_size) {
				int dst_addr = p_code_ptr[p_ip + 1];
				int src_addr = p_code_ptr[p_ip + 2];
				String dst = resolve_address(dst_addr, p_function, true);
				String src = resolve_address(src_addr, p_function);
				opcode_code += vformat("    %s = %s;\n", dst.utf8().get_data(), src.utf8().get_data());
			}
			// Variable size opcodes, advance conservatively
			p_ip += 3; // opcode + src + dst + type info
			break;
		}
		case GDScriptFunction::OPCODE_JUMP: {
			if (p_ip + 1 < p_code_size) {
				int target_ip = p_code_ptr[p_ip + 1];
				HashMap<int, int> jump_labels;
				generate_jump_labels(p_code_ptr, p_code_size, jump_labels);
				int label_id = jump_labels[target_ip];
				opcode_code += vformat("    goto label_%d;\n", label_id);
			}
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_JUMP_IF: {
			if (p_ip + 2 < p_code_size) {
				int condition_addr = p_code_ptr[p_ip + 1];
				int target_ip = p_code_ptr[p_ip + 2];
				String condition = resolve_address(condition_addr, p_function);
				HashMap<int, int> jump_labels;
				generate_jump_labels(p_code_ptr, p_code_size, jump_labels);
				int label_id = jump_labels[target_ip];
				opcode_code += vformat("    if (%s.booleanize()) goto label_%d;\n",
						condition.utf8().get_data(), label_id);
			}
			p_ip += 3; // opcode + condition + target
			break;
		}
		case GDScriptFunction::OPCODE_JUMP_IF_NOT: {
			if (p_ip + 2 < p_code_size) {
				int condition_addr = p_code_ptr[p_ip + 1];
				int target_ip = p_code_ptr[p_ip + 2];
				String condition = resolve_address(condition_addr, p_function);
				HashMap<int, int> jump_labels;
				generate_jump_labels(p_code_ptr, p_code_size, jump_labels);
				int label_id = jump_labels[target_ip];
				opcode_code += vformat("    if (!(%s).booleanize()) goto label_%d;\n",
						condition.utf8().get_data(), label_id);
			}
			p_ip += 3; // opcode + condition + target
			break;
		}
		case GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT: {
			// Default argument handling - skip for C code (handled in VM)
			if (p_ip + 1 < p_code_size) {
				int target_ip = p_code_ptr[p_ip + 1];
				HashMap<int, int> jump_labels;
				generate_jump_labels(p_code_ptr, p_code_size, jump_labels);
				int label_id = jump_labels[target_ip];
				opcode_code += vformat("    goto label_%d;\n", label_id);
			}
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_JUMP_IF_SHARED: {
			// Shared object check - skip for C code (handled in VM)
			if (p_ip + 2 < p_code_size) {
				int value_addr = p_code_ptr[p_ip + 1];
				int target_ip = p_code_ptr[p_ip + 2];
				HashMap<int, int> jump_labels;
				generate_jump_labels(p_code_ptr, p_code_size, jump_labels);
				int label_id = jump_labels[target_ip];
				opcode_code += vformat("    // Shared jump check - goto label_%d;\n", label_id);
			}
			p_ip += 3;
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED: {
			if (p_ip + 4 < p_code_size) {
				int result_addr = p_code_ptr[p_ip + 1];
				int left_addr = p_code_ptr[p_ip + 2];
				int right_addr = p_code_ptr[p_ip + 3];
				int op_index = p_code_ptr[p_ip + 4];

				String result = resolve_address(result_addr, p_function, true);
				String left = resolve_address(left_addr, p_function);
				String right = resolve_address(right_addr, p_function);

				opcode_code += vformat("    {\n");
				opcode_code += vformat("        Variant::ValidatedOperatorEvaluator op_func = operator_funcs[%d];\n", op_index);
				opcode_code += vformat("        op_func(&%s, &%s, &%s);\n",
						left.utf8().get_data(), right.utf8().get_data(), result.utf8().get_data());
				opcode_code += vformat("    }\n");
			}
			p_ip += 5; // opcode + result + left + right + op_index
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR: {
			// Non-validated operator - fall back to VM syscall
			opcode_code += "    // Non-validated operator - TODO: implement\n";
			p_ip += 5; // Skip all arguments
			break;
		}
		case GDScriptFunction::OPCODE_TYPE_TEST_BUILTIN:
		case GDScriptFunction::OPCODE_TYPE_TEST_ARRAY:
		case GDScriptFunction::OPCODE_TYPE_TEST_DICTIONARY:
		case GDScriptFunction::OPCODE_TYPE_TEST_NATIVE:
		case GDScriptFunction::OPCODE_TYPE_TEST_SCRIPT: {
			// Type tests - use VM fallback
			opcode_code += "    // Type test - TODO: implement\n";
			p_ip += 4; // Variable size
			break;
		}
		case GDScriptFunction::OPCODE_SET_KEYED:
		case GDScriptFunction::OPCODE_SET_KEYED_VALIDATED:
		case GDScriptFunction::OPCODE_SET_INDEXED_VALIDATED: {
			// Keyed/Indexed set - use VM fallback
			opcode_code += "    // Set keyed/indexed - TODO: implement\n";
			p_ip += 4; // opcode + target + index + source
			break;
		}
		case GDScriptFunction::OPCODE_GET_KEYED:
		case GDScriptFunction::OPCODE_GET_KEYED_VALIDATED:
		case GDScriptFunction::OPCODE_GET_INDEXED_VALIDATED: {
			// Keyed/Indexed get - use VM fallback
			opcode_code += "    // Get keyed/indexed - TODO: implement\n";
			p_ip += 4; // opcode + target + source + index
			break;
		}
		case GDScriptFunction::OPCODE_SET_NAMED:
		case GDScriptFunction::OPCODE_SET_NAMED_VALIDATED: {
			// Named set - use VM fallback
			opcode_code += "    // Set named - TODO: implement\n";
			p_ip += 3; // opcode + target + source (name embedded)
			break;
		}
		case GDScriptFunction::OPCODE_GET_NAMED:
		case GDScriptFunction::OPCODE_GET_NAMED_VALIDATED: {
			// Named get - use VM fallback
			opcode_code += "    // Get named - TODO: implement\n";
			p_ip += 3; // opcode + target + source (name embedded)
			break;
		}
		case GDScriptFunction::OPCODE_GET_MEMBER: {
			// Property access via syscall
			if (p_ip + 2 < p_code_size) {
				int result_addr = p_code_ptr[p_ip + 1];
				int member_index = p_code_ptr[p_ip + 2];
				String member_name = p_function->get_global_name(member_index);

				String result = resolve_address(result_addr, p_function, true);
				Vector<String> syscall_args;
				syscall_args.push_back("*(Variant*)instance"); // object
				syscall_args.push_back("\"" + member_name + "\""); // property name
				syscall_args.push_back(itos(member_name.length())); // name length
				syscall_args.push_back(result); // result pointer

				opcode_code += generate_syscall(ECALL_OBJ_PROP_GET, syscall_args);
			}
			p_ip += 3; // opcode + result + member_index
			break;
		}
		case GDScriptFunction::OPCODE_SET_MEMBER: {
			// Property assignment via syscall
			if (p_ip + 2 < p_code_size) {
				int value_addr = p_code_ptr[p_ip + 1];
				int member_index = p_code_ptr[p_ip + 2];
				String member_name = p_function->get_global_name(member_index);

				String value = resolve_address(value_addr, p_function);
				Vector<String> syscall_args;
				syscall_args.push_back("*(Variant*)instance"); // object
				syscall_args.push_back("\"" + member_name + "\""); // property name
				syscall_args.push_back(itos(member_name.length())); // name length
				syscall_args.push_back(value); // value

				opcode_code += generate_syscall(ECALL_OBJ_PROP_SET, syscall_args);
			}
			p_ip += 3; // opcode + value + member_index
			break;
		}
		case GDScriptFunction::OPCODE_SET_STATIC_VARIABLE:
		case GDScriptFunction::OPCODE_GET_STATIC_VARIABLE: {
			// Static variables - use VM fallback
			opcode_code += "    // Static variable access - TODO: implement\n";
			p_ip += 4; // opcode + target + class + index
			break;
		}
		case GDScriptFunction::OPCODE_CAST_TO_BUILTIN:
		case GDScriptFunction::OPCODE_CAST_TO_NATIVE:
		case GDScriptFunction::OPCODE_CAST_TO_SCRIPT: {
			// Cast operations - use VM fallback
			opcode_code += "    // Cast operation - TODO: implement\n";
			p_ip += 4; // opcode + source + target + type info
			break;
		}
		case GDScriptFunction::OPCODE_CONSTRUCT:
		case GDScriptFunction::OPCODE_CONSTRUCT_VALIDATED:
		case GDScriptFunction::OPCODE_CONSTRUCT_ARRAY:
		case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_ARRAY:
		case GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY:
		case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_DICTIONARY: {
			// Constructor calls - use VM fallback
			opcode_code += "    // Constructor call - TODO: implement\n";
			p_ip += 4; // Variable size - skip conservatively
			break;
		}
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
		case GDScriptFunction::OPCODE_CALL_METHOD_BIND_VALIDATED_NO_RETURN: {
			// Complex method calls - use VM syscall
			opcode_code += "    // Method call - TODO: implement via syscall\n";
			p_ip += 3; // Variable size - skip conservatively
			break;
		}
		case GDScriptFunction::OPCODE_AWAIT:
		case GDScriptFunction::OPCODE_AWAIT_RESUME: {
			// Await operations - not supported in ELF (must use VM)
			opcode_code += "    // Await - not supported in ELF, use VM\n";
			p_ip += 3; // opcode + operand + resume target
			break;
		}
		case GDScriptFunction::OPCODE_CREATE_LAMBDA:
		case GDScriptFunction::OPCODE_CREATE_SELF_LAMBDA: {
			// Lambda creation - use VM fallback
			opcode_code += "    // Lambda creation - TODO: implement\n";
			p_ip += 4; // Variable size
			break;
		}
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
		case GDScriptFunction::OPCODE_ITERATE_BEGIN_RANGE: {
			// Iteration begin - use VM fallback
			opcode_code += "    // Iteration begin - TODO: implement\n";
			p_ip += 5; // Variable size
			break;
		}
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
		case GDScriptFunction::OPCODE_ITERATE_RANGE: {
			// Iteration step - use VM fallback
			opcode_code += "    // Iteration step - TODO: implement\n";
			p_ip += 4; // Variable size
			break;
		}
		case GDScriptFunction::OPCODE_STORE_GLOBAL:
		case GDScriptFunction::OPCODE_STORE_NAMED_GLOBAL: {
			// Global operations - use VM fallback
			opcode_code += "    // Global access - TODO: implement\n";
			p_ip += 3; // opcode + dst + global_index/name
			break;
		}
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
		case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR4_ARRAY: {
			// Type adjustment - handled in bytecode generation, no C code needed
			opcode_code += vformat("    // Type adjustment %d\n", (int)p_opcode);
			p_ip += 2; // opcode + target
			break;
		}
		case GDScriptFunction::OPCODE_ASSERT: {
			// Assertions - skip in ELF (handled in VM at bytecode level)
			opcode_code += "    // Assertion - handled in VM\n";
			p_ip += 3; // opcode + test + message
			break;
		}
		case GDScriptFunction::OPCODE_BREAKPOINT: {
			// Breakpoints - no C code
			opcode_code += "    // Breakpoint\n";
			p_ip += 1;
			break;
		}
		case GDScriptFunction::OPCODE_LINE: {
			// Debug line info - no C code
			opcode_code += vformat("    // Line %d\n", p_ip + 1 < p_code_size ? p_code_ptr[p_ip + 1] : 0);
			p_ip += 2; // opcode + line number
			break;
		}
		case GDScriptFunction::OPCODE_END: {
			// End of function
			opcode_code += "    // Function end\n";
			break;
		}
		default: {
			// Unknown opcode - fallback
			opcode_code += vformat("    // Unknown opcode %d\n", (int)p_opcode);
			p_ip += 1;
			break;
		}
	}

	return opcode_code;
}

String GDScriptBytecodeCCodeGenerator::generate_c_code(GDScriptFunction *p_function) const {
	if (!p_function || !p_function->_code_ptr || p_function->_code_size == 0) {
		return String();
	}

	String function_name = p_function->get_name();
	bool is_static = p_function->is_static();

	// Generate complete C function
	String code;
	code += "#include <stdint.h>\n";
	code += "#include \"variant.h\"  // Godot Variant type\n\n";

	// Function signature
	code += generate_function_signature(function_name, is_static);
	code += "\n";

	// Function body
	int stack_size = p_function->get_max_stack_size();
	code += generate_prelogue(stack_size, p_function);

	// Generate jump labels
	HashMap<int, int> jump_labels;
	generate_jump_labels(p_function->_code_ptr, p_function->_code_size, jump_labels);

	// Opcodes processing loop
	const int *code_ptr = p_function->_code_ptr;
	int code_size = p_function->_code_size;
	int ip = 0;

	while (ip < code_size) {
		int opcode = code_ptr[ip];

		// Add jump label if this IP is a target
		if (jump_labels.has(ip)) {
			code += vformat("label_%d:\n", jump_labels[ip]);
		}

		String opcode_code = generate_opcode((GDScriptFunction::Opcode)opcode, code_ptr, ip, code_size, p_function);
		code += opcode_code;
	}

	// Epilogue
	code += generate_epilogue();

	return code;
}
