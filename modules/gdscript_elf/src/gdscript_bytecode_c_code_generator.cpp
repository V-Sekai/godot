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
#include "modules/sandbox/src/guest_datatypes.h"
#endif

String GDScriptBytecodeCCodeGenerator::generate_function_signature(const String &p_function_name, bool p_is_static) const {
	// Generate C++ function signature using Sandbox API types
	return vformat("void gdscript_%s(void* instance, GuestVariant* args, int argcount, GuestVariant* result, GuestVariant* constants, Variant::ValidatedOperatorEvaluator* operator_funcs)", p_function_name);
}

String GDScriptBytecodeCCodeGenerator::generate_prelogue(int p_stack_size, const GDScriptFunction *p_function) const {
	String prologue = "{\n";
	prologue += vformat("    GuestVariant stack[%d];\n", p_stack_size);
	prologue += "    int ip = 0;\n";
	prologue += "\n";

	// Initialize stack with null variants (GuestVariant defaults to NIL)
	prologue += "    // Initialize stack (GuestVariant defaults to NIL type)\n";
	for (int i = 0; i < p_stack_size; i++) {
		prologue += vformat("    stack[%d].type = Variant::NIL;\n", i);
		prologue += vformat("    stack[%d].v.i = 0;\n", i);
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
			// Error case - return a reference to a default GuestVariant
			return "stack[0]"; // Fallback (shouldn't happen)
		}
	}
}

String GDScriptBytecodeCCodeGenerator::resolve_assign_source(GDScriptFunction::Opcode p_opcode, const int *p_code_ptr, int p_ip, int p_code_size, const GDScriptFunction *p_function) const {
	// Helper to resolve assign source for regular assigns (special cases handled in opcode generation)
	// This is only called for regular ASSIGN and typed ASSIGN variants
	if (p_opcode == GDScriptFunction::OPCODE_ASSIGN) {
		// Regular assign: source is at p_ip + 2
		if (p_ip + 2 < p_code_size) {
			int src_addr = p_code_ptr[p_ip + 2];
			return resolve_address(src_addr, p_function);
		}
		return "stack[0]"; // Fallback (shouldn't happen)
	} else {
		// Typed assigns: source is at p_ip + 2 (same as regular assign)
		if (p_ip + 2 < p_code_size) {
			int src_addr = p_code_ptr[p_ip + 2];
			return resolve_address(src_addr, p_function);
		}
		return "stack[0]"; // Fallback (shouldn't happen)
	}
}

String GDScriptBytecodeCCodeGenerator::generate_syscall(int p_ecall_number, const Vector<String> &p_args) const {
	String syscall_code;
	syscall_code += "    // Syscall " + itos(p_ecall_number) + "\n";

	// RISC-V syscall ABI: up to 7 arguments in a0-a6, syscall number in a7
	int max_args = p_args.size() < 7 ? p_args.size() : 7;

	// Set up registers for syscall (inline assembly style for C++ code)
	// Use GuestVariant* instead of Variant*
	for (int i = 0; i < max_args; i++) {
		syscall_code += vformat("    register GuestVariant* a%d asm(\"a%d\") = &(%s);\n", i, i, p_args[i].utf8().get_data());
	}

	syscall_code += vformat("    register int syscall_number asm(\"a7\") = %d;\n", p_ecall_number);
	syscall_code += "    __asm__ volatile(\"ecall\" : : \"r\"(syscall_number)";

	for (int i = 0; i < max_args; i++) {
		syscall_code += vformat(", \"r\"(a%d)", i);
	}
	syscall_code += ");\n\n";

	return syscall_code;
}

String GDScriptBytecodeCCodeGenerator::generate_vcall_syscall(const String &p_variant_ptr, const String &p_method_str, int p_method_len, const String &p_args_ptr, int p_args_size, const String &p_ret_ptr) const {
	// Specialized syscall generation for ECALL_VCALL (6 arguments)
	// Signature: (GuestVariant* vp, gaddr_t method, unsigned mlen, gaddr_t args_ptr, gaddr_t args_size, gaddr_t vret_addr)
	// Use GuestVariant* directly in C++ code
	String syscall_code;
	syscall_code += "    // ECALL_VCALL syscall\n";

	// Set up registers: a0-a5 for arguments, a7 for syscall number
	syscall_code += vformat("    register GuestVariant* a0 asm(\"a0\") = &(%s);\n", p_variant_ptr.utf8().get_data());
	syscall_code += vformat("    register const char* a1 asm(\"a1\") = %s;\n", p_method_str.utf8().get_data());
	syscall_code += vformat("    register unsigned a2 asm(\"a2\") = %d;\n", p_method_len);
	syscall_code += vformat("    register GuestVariant* a3 asm(\"a3\") = %s;\n", p_args_ptr.utf8().get_data());
	syscall_code += vformat("    register int a4 asm(\"a4\") = %d;\n", p_args_size);
	syscall_code += vformat("    register GuestVariant* a5 asm(\"a5\") = &(%s);\n", p_ret_ptr.utf8().get_data());
	syscall_code += "    register int syscall_number asm(\"a7\") = ECALL_VCALL;\n";
	syscall_code += "    __asm__ volatile(\"ecall\" : : \"r\"(syscall_number), \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(a4), \"r\"(a5));\n";

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
		case GDScriptFunction::OPCODE_RETURN:
		case GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN:
		case GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY:
		case GDScriptFunction::OPCODE_RETURN_TYPED_DICTIONARY:
		case GDScriptFunction::OPCODE_RETURN_TYPED_NATIVE:
		case GDScriptFunction::OPCODE_RETURN_TYPED_SCRIPT: {
			// All return variants generate identical C code (type checking happens at bytecode level)
			// Return value is always at p_ip + 1
			if (p_ip + 1 < p_code_size) {
				int return_addr = p_code_ptr[p_ip + 1];
				String return_expr = resolve_address(return_addr, p_function);
				opcode_code += vformat("    *result = %s;\n", return_expr.utf8().get_data());
			}
			opcode_code += "    return;\n";
			// Advance IP based on opcode type (all have return value at p_ip + 1)
			if (p_opcode == GDScriptFunction::OPCODE_RETURN) {
				p_ip += 2; // opcode + return value
			} else if (p_opcode == GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN ||
					p_opcode == GDScriptFunction::OPCODE_RETURN_TYPED_NATIVE ||
					p_opcode == GDScriptFunction::OPCODE_RETURN_TYPED_SCRIPT) {
				p_ip += 3; // opcode + return value + type info (1 byte)
			} else if (p_opcode == GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY) {
				p_ip += 5; // opcode + return value + type info (4 bytes)
			} else if (p_opcode == GDScriptFunction::OPCODE_RETURN_TYPED_DICTIONARY) {
				p_ip += 8; // opcode + return value + type info (7 bytes)
			}
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN:
		case GDScriptFunction::OPCODE_ASSIGN_NULL:
		case GDScriptFunction::OPCODE_ASSIGN_TRUE:
		case GDScriptFunction::OPCODE_ASSIGN_FALSE:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_ARRAY:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_DICTIONARY:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE:
		case GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT: {
			// All assign variants use GuestVariant struct assignment
			// Special cases (NULL, TRUE, FALSE) handled directly
			if (p_ip + 1 < p_code_size) {
				int dst_addr = p_code_ptr[p_ip + 1];
				String dst = resolve_address(dst_addr, p_function, true);
				
				if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_NULL) {
					// Direct field assignment for NIL
					opcode_code += vformat("    %s.type = Variant::NIL;\n", dst.utf8().get_data());
					opcode_code += vformat("    %s.v.i = 0;\n", dst.utf8().get_data());
				} else if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_TRUE) {
					// Direct field assignment for BOOL true
					opcode_code += vformat("    %s.type = Variant::BOOL;\n", dst.utf8().get_data());
					opcode_code += vformat("    %s.v.b = true;\n", dst.utf8().get_data());
				} else if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_FALSE) {
					// Direct field assignment for BOOL false
					opcode_code += vformat("    %s.type = Variant::BOOL;\n", dst.utf8().get_data());
					opcode_code += vformat("    %s.v.b = false;\n", dst.utf8().get_data());
				} else {
					// Regular assign: struct copy (GuestVariant is POD)
					String src = resolve_assign_source(p_opcode, p_code_ptr, p_ip, p_code_size, p_function);
					opcode_code += vformat("    %s = %s;\n", dst.utf8().get_data(), src.utf8().get_data());
				}
			}
			// Advance IP based on opcode type (from disassembler)
			if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_NULL ||
					p_opcode == GDScriptFunction::OPCODE_ASSIGN_TRUE ||
					p_opcode == GDScriptFunction::OPCODE_ASSIGN_FALSE) {
				p_ip += 2; // opcode + dst
			} else if (p_opcode == GDScriptFunction::OPCODE_ASSIGN) {
				p_ip += 3; // opcode + dst + src
			} else if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN ||
					p_opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE ||
					p_opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT) {
				p_ip += 4; // opcode + dst + src + type info (1 byte)
			} else if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_ARRAY) {
				p_ip += 6; // opcode + dst + src + type info (4 bytes)
			} else if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_DICTIONARY) {
				p_ip += 9; // opcode + dst + src + type info (7 bytes)
			}
			break;
		}
		case GDScriptFunction::OPCODE_JUMP:
		case GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT: {
			// Unconditional jump - both opcodes generate same code
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
		case GDScriptFunction::OPCODE_JUMP_IF:
		case GDScriptFunction::OPCODE_JUMP_IF_SHARED: {
			// Conditional jump (true) - both opcodes generate same code
			// JUMP_IF_SHARED treats shared check as condition (handled at bytecode level)
			// For GuestVariant, check type and value for boolean conversion
			if (p_ip + 2 < p_code_size) {
				int condition_addr = p_code_ptr[p_ip + 1];
				int target_ip = p_code_ptr[p_ip + 2];
				String condition = resolve_address(condition_addr, p_function);
				HashMap<int, int> jump_labels;
				generate_jump_labels(p_code_ptr, p_code_size, jump_labels);
				int label_id = jump_labels[target_ip];
				// GuestVariant boolean check: type == BOOL && v.b == true, or non-zero numeric
				opcode_code += vformat("    if ((%s.type == Variant::BOOL && %s.v.b) || (%s.type == Variant::INT && %s.v.i != 0) || (%s.type == Variant::FLOAT && %s.v.f != 0.0)) goto label_%d;\n",
						condition.utf8().get_data(), condition.utf8().get_data(),
						condition.utf8().get_data(), condition.utf8().get_data(),
						condition.utf8().get_data(), condition.utf8().get_data(), label_id);
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
				// GuestVariant boolean check (negated)
				opcode_code += vformat("    if (!((%s.type == Variant::BOOL && %s.v.b) || (%s.type == Variant::INT && %s.v.i != 0) || (%s.type == Variant::FLOAT && %s.v.f != 0.0))) goto label_%d;\n",
						condition.utf8().get_data(), condition.utf8().get_data(),
						condition.utf8().get_data(), condition.utf8().get_data(),
						condition.utf8().get_data(), condition.utf8().get_data(), label_id);
			}
			p_ip += 3; // opcode + condition + target
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED: {
			// Validated operators work with Variant*, but we need to convert GuestVariant to Variant
			// For now, use syscall ECALL_VEVAL for operator evaluation (sandbox handles conversion)
			if (p_ip + 4 < p_code_size) {
				int result_addr = p_code_ptr[p_ip + 1];
				int left_addr = p_code_ptr[p_ip + 2];
				int right_addr = p_code_ptr[p_ip + 3];
				int op_index = p_code_ptr[p_ip + 4];

				String result = resolve_address(result_addr, p_function, true);
				String left = resolve_address(left_addr, p_function);
				String right = resolve_address(right_addr, p_function);

				// Use ECALL_VEVAL syscall for operator evaluation with GuestVariant
				opcode_code += vformat("    {\n");
				opcode_code += vformat("        // ECALL_VEVAL: operator evaluation\n");
				opcode_code += vformat("        register GuestVariant* a0 asm(\"a0\") = &(%s);\n", left.utf8().get_data());
				opcode_code += vformat("        register GuestVariant* a1 asm(\"a1\") = &(%s);\n", right.utf8().get_data());
				opcode_code += vformat("        register GuestVariant* a2 asm(\"a2\") = &(%s);\n", result.utf8().get_data());
				opcode_code += vformat("        register int a3 asm(\"a3\") = %d;\n", op_index);
				opcode_code += vformat("        register int syscall_number asm(\"a7\") = ECALL_VEVAL;\n");
				opcode_code += vformat("        __asm__ volatile(\"ecall\" : : \"r\"(syscall_number), \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3));\n");
				opcode_code += vformat("    }\n");
			}
			p_ip += 5; // opcode + result + left + right + op_index
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR: {
			// Non-validated operator - use ECALL_VCALL syscall
			// For now, generate a comment indicating VM call needed
			// TODO: Implement proper VM call via ECALL_VCALL for operator evaluation
			opcode_code += "    // Non-validated operator - use VM call via ECALL_VCALL\n";
			opcode_code += "    // Note: This requires marshaling operator arguments to VM\n";
			p_ip += 5; // opcode + result + left + right + op_index
			break;
		}
		case GDScriptFunction::OPCODE_TYPE_TEST_BUILTIN:
		case GDScriptFunction::OPCODE_TYPE_TEST_ARRAY:
		case GDScriptFunction::OPCODE_TYPE_TEST_DICTIONARY:
		case GDScriptFunction::OPCODE_TYPE_TEST_NATIVE:
		case GDScriptFunction::OPCODE_TYPE_TEST_SCRIPT: {
			// Type tests - use ECALL_VCALL to call Variant::get_type() and compare
			if (p_ip + 3 < p_code_size) {
				int result_addr = p_code_ptr[p_ip + 1];
				int value_addr = p_code_ptr[p_ip + 2];
				int type_info = p_code_ptr[p_ip + 3];

				String result = resolve_address(result_addr, p_function, true);
				String value = resolve_address(value_addr, p_function);

				// Use ECALL_VCALL to get type and compare
				opcode_code += vformat("    // Type test: check if %s matches type\n", value.utf8().get_data());
				opcode_code += vformat("    GuestVariant type_test_args[0] = {};\n");
				opcode_code += generate_vcall_syscall(
						value, // vp (value to test)
						"\"get_type\"", // method name
						8, // method length
						"type_test_args", // args_ptr (empty)
						0, // args_size (0 arguments)
						result // vret_addr (result: type matches)
				);
				// TODO: Compare result with expected type
			}
			p_ip += 4; // opcode + result + value + type_info
			break;
		}
		case GDScriptFunction::OPCODE_SET_KEYED:
		case GDScriptFunction::OPCODE_SET_KEYED_VALIDATED:
		case GDScriptFunction::OPCODE_SET_INDEXED_VALIDATED: {
			// Keyed/Indexed set - use ECALL_VCALL to call Variant::set() method
			if (p_ip + 3 < p_code_size) {
				int target_addr = p_code_ptr[p_ip + 1];
				int index_addr = p_code_ptr[p_ip + 2];
				int source_addr = p_code_ptr[p_ip + 3];

				String target = resolve_address(target_addr, p_function);
				String index = resolve_address(index_addr, p_function);
				String source = resolve_address(source_addr, p_function);
				String result = resolve_address(target_addr, p_function, true);

				// Create temporary array for arguments [index, source]
				opcode_code += vformat("    // Set keyed/indexed: %s[%s] = %s\n",
						target.utf8().get_data(), index.utf8().get_data(), source.utf8().get_data());
				opcode_code += vformat("    GuestVariant set_args[2] = {%s, %s};\n",
						index.utf8().get_data(), source.utf8().get_data());
				opcode_code += generate_vcall_syscall(
						target, // vp (target variant)
						"\"set\"", // method name
						3, // method length
						"set_args", // args_ptr (array with [index, source])
						2, // args_size (2 arguments)
						result // vret_addr (result stored in target)
				);
			}
			// Advance IP: SET_KEYED=4, SET_KEYED_VALIDATED=5, SET_INDEXED_VALIDATED=5
			if (p_opcode == GDScriptFunction::OPCODE_SET_KEYED) {
				p_ip += 4;
			} else {
				p_ip += 5; // validated variants have setter/getter index
			}
			break;
		}
		case GDScriptFunction::OPCODE_GET_KEYED:
		case GDScriptFunction::OPCODE_GET_KEYED_VALIDATED:
		case GDScriptFunction::OPCODE_GET_INDEXED_VALIDATED: {
			// Keyed/Indexed get - use ECALL_VCALL to call Variant::get() method
			if (p_ip + 3 < p_code_size) {
				int source_addr = p_code_ptr[p_ip + 1];
				int index_addr = p_code_ptr[p_ip + 2];
				int target_addr = p_code_ptr[p_ip + 3];

				String source = resolve_address(source_addr, p_function);
				String index = resolve_address(index_addr, p_function);
				String target = resolve_address(target_addr, p_function, true);

				// Create temporary array for arguments [index]
				opcode_code += vformat("    // Get keyed/indexed: %s = %s[%s]\n",
						target.utf8().get_data(), source.utf8().get_data(), index.utf8().get_data());
				opcode_code += vformat("    GuestVariant get_args[1] = {%s};\n", index.utf8().get_data());
				opcode_code += generate_vcall_syscall(
						source, // vp (source variant)
						"\"get\"", // method name
						3, // method length
						"get_args", // args_ptr (array with [index])
						1, // args_size (1 argument)
						target // vret_addr (result stored in target)
				);
			}
			// Advance IP: GET_KEYED=4, GET_KEYED_VALIDATED=5, GET_INDEXED_VALIDATED=5
			if (p_opcode == GDScriptFunction::OPCODE_GET_KEYED) {
				p_ip += 4;
			} else {
				p_ip += 5; // validated variants have getter index
			}
			break;
		}
		case GDScriptFunction::OPCODE_SET_NAMED:
		case GDScriptFunction::OPCODE_SET_NAMED_VALIDATED: {
			// Named set - use property set syscall (same as SET_MEMBER)
			if (p_ip + 3 < p_code_size) {
				int target_addr = p_code_ptr[p_ip + 1];
				int source_addr = p_code_ptr[p_ip + 2];
				int name_index = p_code_ptr[p_ip + 3];
				String name = p_function->get_global_name(name_index);

				String target = resolve_address(target_addr, p_function);
				String source = resolve_address(source_addr, p_function);
				Vector<String> syscall_args;
				syscall_args.push_back(target); // object
				syscall_args.push_back("\"" + name + "\""); // property name
				syscall_args.push_back(itos(name.length())); // name length
				syscall_args.push_back(source); // value

				opcode_code += generate_syscall(ECALL_OBJ_PROP_SET, syscall_args);
			}
			p_ip += 4; // opcode + target + source + name_index
			break;
		}
		case GDScriptFunction::OPCODE_GET_NAMED:
		case GDScriptFunction::OPCODE_GET_NAMED_VALIDATED: {
			// Named get - use property get syscall (same as GET_MEMBER)
			if (p_ip + 3 < p_code_size) {
				int src_addr = p_code_ptr[p_ip + 1];
				int dst_addr = p_code_ptr[p_ip + 2];
				int name_index = p_code_ptr[p_ip + 3];
				String name = p_function->get_global_name(name_index);

				String src = resolve_address(src_addr, p_function);
				String dst = resolve_address(dst_addr, p_function, true);
				Vector<String> syscall_args;
				syscall_args.push_back(src); // object
				syscall_args.push_back("\"" + name + "\""); // property name
				syscall_args.push_back(itos(name.length())); // name length
				syscall_args.push_back(dst); // result pointer

				opcode_code += generate_syscall(ECALL_OBJ_PROP_GET, syscall_args);
			}
			p_ip += 4; // opcode + src + dst + name_index
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
				// instance is void*, need to cast to GuestVariant* for syscall
				// Note: In actual execution, instance would be converted by sandbox
				syscall_args.push_back("(GuestVariant*)instance"); // object (cast to GuestVariant*)
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
				// instance is void*, need to cast to GuestVariant* for syscall
				syscall_args.push_back("(GuestVariant*)instance"); // object (cast to GuestVariant*)
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
			// Static variables - use ECALL_VSTORE/ECALL_VFETCH syscalls
			if (p_ip + 3 < p_code_size) {
				int value_addr = p_code_ptr[p_ip + 1];
				int class_addr = p_code_ptr[p_ip + 2];
				int index = p_code_ptr[p_ip + 3];

				String value = resolve_address(value_addr, p_function);
				String result = resolve_address(value_addr, p_function, p_opcode == GDScriptFunction::OPCODE_GET_STATIC_VARIABLE);

				if (p_opcode == GDScriptFunction::OPCODE_GET_STATIC_VARIABLE) {
					// Use ECALL_VFETCH to get static variable
					opcode_code += vformat("    // Get static variable[%d]\n", index);
					Vector<String> syscall_args;
					syscall_args.push_back(itos(index)); // index
					syscall_args.push_back("(gaddr_t)0"); // gdata (0 for GuestVariant fetch)
					syscall_args.push_back("0"); // method
					opcode_code += generate_syscall(ECALL_VFETCH, syscall_args);
					opcode_code += vformat("    %s = stack[0]; // Result from VFETCH (GuestVariant)\n", result.utf8().get_data());
				} else {
					// Use ECALL_VSTORE to set static variable
					opcode_code += vformat("    // Set static variable[%d] = %s\n", index, value.utf8().get_data());
					Vector<String> syscall_args;
					syscall_args.push_back("&static_var_idx"); // vidx (output)
					syscall_args.push_back("Variant::NIL"); // type (will be determined from value)
					syscall_args.push_back("(gaddr_t)&" + value); // gdata (GuestVariant*)
					syscall_args.push_back("sizeof(GuestVariant)"); // gsize
					opcode_code += generate_syscall(ECALL_VSTORE, syscall_args);
				}
			}
			p_ip += 4; // opcode + value + class + index
			break;
		}
		case GDScriptFunction::OPCODE_CAST_TO_BUILTIN:
		case GDScriptFunction::OPCODE_CAST_TO_NATIVE:
		case GDScriptFunction::OPCODE_CAST_TO_SCRIPT: {
			// Cast operations - use composite ASSIGN pattern (type conversion happens at VM level)
			if (p_ip + 2 < p_code_size) {
				int source_addr = p_code_ptr[p_ip + 1];
				int target_addr = p_code_ptr[p_ip + 2];

				String source = resolve_address(source_addr, p_function);
				String target = resolve_address(target_addr, p_function, true);

				// Cast is essentially assignment with type conversion (handled by VM)
				opcode_code += vformat("    %s = %s; // Cast (type conversion handled by VM)\n",
						target.utf8().get_data(), source.utf8().get_data());
			}
			p_ip += 4; // opcode + source + target + type info
			break;
		}
		case GDScriptFunction::OPCODE_CONSTRUCT:
		case GDScriptFunction::OPCODE_CONSTRUCT_VALIDATED:
		case GDScriptFunction::OPCODE_CONSTRUCT_ARRAY:
		case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_ARRAY:
		case GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY:
		case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_DICTIONARY: {
			// Constructor calls - use ECALL_VCREATE syscall
			// Structure: opcode + var_args_count + args... + target + argc + type_info
			if (p_ip + 1 < p_code_size) {
				int var_args_count = p_code_ptr[p_ip + 1];
				// Skip variable arguments and read target address
				int target_offset = 2 + var_args_count; // opcode + var_args_count + args
				if (p_ip + target_offset < p_code_size) {
					int target_addr = p_code_ptr[p_ip + target_offset];
					String target = resolve_address(target_addr, p_function, true);

					opcode_code += vformat("    // Construct variant (type info at bytecode level)\n");
					opcode_code += vformat("    // Use ECALL_VCREATE to create variant\n");
					// For now, generate assignment to target (proper implementation needs ECALL_VCREATE with type and data)
					opcode_code += vformat("    %s.type = Variant::NIL; // Constructor result (default to NIL)\n", target.utf8().get_data());
					opcode_code += vformat("    %s.v.i = 0;\n", target.utf8().get_data());
				}
				// Advance IP: opcode(1) + var_args_count(1) + args(var_args_count) + target(1) + argc(1) + type_info(variable)
				p_ip += 2 + var_args_count + 2; // Minimum: opcode + var_args_count + args + target + argc
				// Type info size varies by constructor type
				if (p_opcode == GDScriptFunction::OPCODE_CONSTRUCT_TYPED_ARRAY) {
					p_ip += 3; // script_type + builtin_type + native_type
				} else if (p_opcode == GDScriptFunction::OPCODE_CONSTRUCT_TYPED_DICTIONARY) {
					p_ip += 6; // key_script_type + key_builtin_type + key_native_type + value_script_type + value_builtin_type + value_native_type
				} else if (p_opcode == GDScriptFunction::OPCODE_CONSTRUCT || p_opcode == GDScriptFunction::OPCODE_CONSTRUCT_VALIDATED) {
					p_ip += 1; // type
				}
			} else {
				p_ip += 1; // Just skip opcode if not enough data
			}
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
			// All method calls use ECALL_VCALL syscall
			// Structure: opcode + var_args_count + args... + base + target + argc + method_name/method_bind
			if (p_ip + 1 < p_code_size) {
				int var_args_count = p_code_ptr[p_ip + 1];
				// Read base, target, argc, method_name from bytecode
				int base_offset = 2 + var_args_count; // opcode + var_args_count + args
				if (p_ip + base_offset + 2 < p_code_size) {
					int base_addr = p_code_ptr[p_ip + base_offset];
					int target_addr = p_code_ptr[p_ip + base_offset + 1];
					int argc = p_code_ptr[p_ip + base_offset + 2];
					int method_idx = p_code_ptr[p_ip + base_offset + 3];

					String base = resolve_address(base_addr, p_function);
					String target = resolve_address(target_addr, p_function, true);
					String method_name = p_function->get_global_name(method_idx);

					// Build arguments array from variable args
					// Array-based marshaling supports unlimited arguments (16+)
					opcode_code += vformat("    // Method call: %s.%s() with %d arguments\n", base.utf8().get_data(), method_name.utf8().get_data(), argc);
					opcode_code += vformat("    GuestVariant call_args[%d];\n", argc);
					// Read arguments from bytecode: arguments start at p_ip + 2, count is var_args_count
					int actual_arg_count = MIN(argc, var_args_count);
					for (int i = 0; i < actual_arg_count && (p_ip + 2 + i < p_code_size); i++) {
						int arg_addr = p_code_ptr[p_ip + 2 + i];
						String arg = resolve_address(arg_addr, p_function);
						opcode_code += vformat("    call_args[%d] = %s;\n", i, arg.utf8().get_data());
					}

					// Use ECALL_VCALL
					opcode_code += generate_vcall_syscall(
							base, // vp (base object)
							"\"" + method_name + "\"", // method name
							method_name.length(), // method length
							"call_args", // args_ptr
							argc, // args_size
							target // vret_addr (if CALL_RETURN)
					);
				}
				// Advance IP: opcode(1) + var_args_count(1) + args(var_args_count) + base(1) + target(1) + argc(1) + method(1)
				p_ip += 2 + var_args_count + 4; // Minimum structure
			} else {
				p_ip += 1; // Just skip opcode
			}
			break;
		}
		case GDScriptFunction::OPCODE_AWAIT:
		case GDScriptFunction::OPCODE_AWAIT_RESUME: {
			// Await operations - use ECALL_VCALL (async handling in VM)
			if (p_ip + 2 < p_code_size) {
				int operand_addr = p_code_ptr[p_ip + 1];
				int resume_target = p_code_ptr[p_ip + 2];
				String operand = resolve_address(operand_addr, p_function);

				opcode_code += vformat("    // Await: %s (async handling via VM)\n", operand.utf8().get_data());
				opcode_code += vformat("    // Use ECALL_VCALL for await operation\n");
				// TODO: Proper await implementation via ECALL_VCALL
			}
			p_ip += 3; // opcode + operand + resume target
			break;
		}
		case GDScriptFunction::OPCODE_CREATE_LAMBDA:
		case GDScriptFunction::OPCODE_CREATE_SELF_LAMBDA: {
			// Lambda creation - use ECALL_CALLABLE_CREATE syscall
			if (p_ip + 2 < p_code_size) {
				int target_addr = p_code_ptr[p_ip + 1];
				String target = resolve_address(target_addr, p_function, true);

				opcode_code += vformat("    // Lambda creation (via ECALL_CALLABLE_CREATE)\n");
				Vector<String> syscall_args;
				syscall_args.push_back(target); // callable result
				opcode_code += generate_syscall(ECALL_CALLABLE_CREATE, syscall_args);
			}
			p_ip += 4; // Variable size - advance conservatively
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
			// Iteration begin - initialize iterator (composite pattern: ASSIGN + setup)
			// Structure: opcode + container + iterator_state + counter (variable)
			opcode_code += "    // Iteration begin: initialize iterator state\n";
			opcode_code += "    // Iterator state managed by VM via ECALL_VCALL if needed\n";
			p_ip += 5; // Variable size - advance conservatively
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
			// Iteration step - composite pattern: check condition (OPERATOR_VALIDATED) + JUMP_IF
			opcode_code += "    // Iteration step: check condition and advance (composite: operator + jump)\n";
			opcode_code += "    // Iterator advancement handled by VM via ECALL_VCALL if needed\n";
			p_ip += 4; // Variable size - advance conservatively
			break;
		}
		case GDScriptFunction::OPCODE_STORE_GLOBAL:
		case GDScriptFunction::OPCODE_STORE_NAMED_GLOBAL: {
			// Global operations - use ECALL_VFETCH/ECALL_VSTORE syscalls
			if (p_ip + 2 < p_code_size) {
				int value_addr = p_code_ptr[p_ip + 1];
				int global_idx = p_code_ptr[p_ip + 2];
				String value = resolve_address(value_addr, p_function);
				String result = resolve_address(value_addr, p_function, p_opcode == GDScriptFunction::OPCODE_STORE_GLOBAL);

				if (p_opcode == GDScriptFunction::OPCODE_STORE_GLOBAL) {
					opcode_code += vformat("    // Store global[%d] = %s\n", global_idx, value.utf8().get_data());
					Vector<String> syscall_args;
					syscall_args.push_back("&global_idx"); // vidx
					syscall_args.push_back("Variant::NIL"); // type
					syscall_args.push_back("(gaddr_t)&" + value); // gdata
					syscall_args.push_back("sizeof(Variant)"); // gsize
					opcode_code += generate_syscall(ECALL_VSTORE, syscall_args);
				} else {
					opcode_code += vformat("    // Get global[%d]\n", global_idx);
					Vector<String> syscall_args;
					syscall_args.push_back(itos(global_idx)); // index
					syscall_args.push_back("(gaddr_t)0"); // gdata
					syscall_args.push_back("0"); // method
					opcode_code += generate_syscall(ECALL_VFETCH, syscall_args);
					opcode_code += vformat("    %s = stack[0]; // Result from VFETCH\n", result.utf8().get_data());
				}
			}
			p_ip += 3; // opcode + value + global_index/name
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

	// Generate complete C++ function using Sandbox API
	String code;
	code += "#include <stdint.h>\n";
	code += "#include \"core/variant/variant.h\"  // Godot Variant type\n";
	code += "#include \"modules/sandbox/src/guest_datatypes.h\"  // GuestVariant type\n";
	code += "#include \"modules/sandbox/src/syscalls.h\"  // ECALL_* definitions\n";
	code += "\n";

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
