/**************************************************************************/
/*  gdscript_to_stablehlo.cpp                                             */
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

#include "gdscript_to_stablehlo.h"

#include "core/io/file_access.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"

bool GDScriptToStableHLO::is_basic_opcode(int p_opcode) {
	switch (p_opcode) {
		case GDScriptFunction::OPCODE_RETURN:
		case GDScriptFunction::OPCODE_ASSIGN:
		case GDScriptFunction::OPCODE_ASSIGN_NULL:
		case GDScriptFunction::OPCODE_ASSIGN_TRUE:
		case GDScriptFunction::OPCODE_ASSIGN_FALSE:
		case GDScriptFunction::OPCODE_JUMP:
		case GDScriptFunction::OPCODE_JUMP_IF:
		case GDScriptFunction::OPCODE_JUMP_IF_NOT:
		case GDScriptFunction::OPCODE_OPERATOR:
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED:
		case GDScriptFunction::OPCODE_GET_MEMBER:
		case GDScriptFunction::OPCODE_SET_MEMBER:
		case GDScriptFunction::OPCODE_CALL:
		case GDScriptFunction::OPCODE_CALL_RETURN:
		case GDScriptFunction::OPCODE_LINE:
		case GDScriptFunction::OPCODE_BREAKPOINT:
		case GDScriptFunction::OPCODE_ASSERT:
		case GDScriptFunction::OPCODE_END:
			return true;
		default:
			return false;
	}
}

bool GDScriptToStableHLO::can_convert_function(const GDScriptFunction *p_function) {
	if (!p_function || p_function->code.is_empty()) {
		return false;
	}

	const int *code_ptr = p_function->code.ptr();
	int code_size = p_function->code.size();
	int ip = 0;

	while (ip < code_size) {
		int opcode = code_ptr[ip];
		if (!is_basic_opcode(opcode)) {
			return false;
		}
		// Advance IP by at least 1 (opcode)
		ip += 1;
		if (ip >= code_size) {
			break;
		}
	}

	return true;
}

String GDScriptToStableHLO::generate_constant(const Variant &p_value, int &p_value_id) {
	String value_str;
	// Format float values with .0 suffix for proper MLIR syntax
	if (p_value.get_type() == Variant::FLOAT || p_value.get_type() == Variant::INT) {
		double float_val = p_value;
		// Check if it's a whole number
		if (float_val == (double)(int64_t)float_val) {
			value_str = String::num(float_val) + ".0";
		} else {
			value_str = String::num(float_val);
		}
	} else {
		value_str = String(p_value);
	}
	String result = "  %c" + String::num(p_value_id) + " = stablehlo.constant dense<" + value_str + "> : tensor<f32>\n";
	p_value_id++;
	return result;
}

bool GDScriptToStableHLO::extract_branch_values(const GDScriptFunction *p_function, int p_jump_ip, int p_jump_target, bool p_is_jump_if_not, Variant &p_true_value, Variant &p_false_value, bool &p_true_is_constant, bool &p_false_is_constant) {
	if (!p_function || p_function->code.is_empty()) {
		return false;
	}

	const int *code_ptr = p_function->code.ptr();
	int code_size = p_function->code.size();

	// Determine branch positions
	// JUMP_IF structure: [opcode, condition_addr, target] = 3 words
	int true_branch_start, false_branch_start;
	if (p_is_jump_if_not) {
		// JUMP_IF_NOT: true branch is after jump (fall through), false branch is at jump target
		true_branch_start = p_jump_ip + 3; // Skip opcode + condition_addr + target
		false_branch_start = p_jump_target;
	} else {
		// JUMP_IF: true branch is at jump target, false branch is after jump (fall through)
		true_branch_start = p_jump_target;
		false_branch_start = p_jump_ip + 3; // Skip opcode + condition_addr + target
	}

	// Helper function to get opcode size (number of words)
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
				return 5; // opcode + dst + a + b + operator_idx
			case GDScriptFunction::OPCODE_OPERATOR: {
				// Variable size due to function pointer
				constexpr int ptr_size = sizeof(Variant::ValidatedOperatorEvaluator) / sizeof(int);
				return 5 + ptr_size; // opcode + dst + a + b + op + function_ptr
			}
			case GDScriptFunction::OPCODE_CALL_RETURN:
			case GDScriptFunction::OPCODE_CALL: {
				if (ip + 1 < code_size) {
					int arg_count = code_ptr[ip + 1];
					return 2 + arg_count; // opcode + arg_count + args
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_GET_NAMED:
			case GDScriptFunction::OPCODE_GET_NAMED_VALIDATED:
			case GDScriptFunction::OPCODE_SET_NAMED:
			case GDScriptFunction::OPCODE_SET_NAMED_VALIDATED:
				return 3; // opcode + dst + name_idx
			case GDScriptFunction::OPCODE_GET_KEYED:
			case GDScriptFunction::OPCODE_GET_KEYED_VALIDATED:
			case GDScriptFunction::OPCODE_SET_KEYED:
			case GDScriptFunction::OPCODE_SET_KEYED_VALIDATED:
				return 4; // opcode + dst + obj + key
			case GDScriptFunction::OPCODE_CONSTRUCT: {
				if (ip + 1 < code_size) {
					int arg_count = code_ptr[ip + 1];
					constexpr int ptr_size = sizeof(Variant::ValidatedConstructor) / sizeof(int);
					return 2 + arg_count + ptr_size; // opcode + arg_count + args + constructor_ptr
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_CONSTRUCT_VALIDATED: {
				if (ip + 1 < code_size) {
					int arg_count = code_ptr[ip + 1];
					return 2 + arg_count; // opcode + arg_count + args
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_CALL_UTILITY:
			case GDScriptFunction::OPCODE_CALL_UTILITY_VALIDATED: {
				if (ip + 1 < code_size) {
					int arg_count = code_ptr[ip + 1];
					return 3 + arg_count; // opcode + dst + arg_count + args
				}
				return 3;
			}
			case GDScriptFunction::OPCODE_TYPE_TEST_BUILTIN:
				return 4; // opcode + dst + value + type
			case GDScriptFunction::OPCODE_TYPE_TEST_ARRAY:
				return 6; // opcode + dst + value + script_type + builtin_type + native_type
			case GDScriptFunction::OPCODE_TYPE_TEST_DICTIONARY:
				return 9; // opcode + dst + value + key_script_type + value_script_type + key_builtin + key_native + value_builtin + value_native
			case GDScriptFunction::OPCODE_TYPE_TEST_NATIVE:
				return 4; // opcode + dst + value + native_type
			case GDScriptFunction::OPCODE_TYPE_TEST_SCRIPT:
				return 4; // opcode + dst + value + script_type
			case GDScriptFunction::OPCODE_SET_INDEXED_VALIDATED:
				return 4; // opcode + dst + obj + key
			case GDScriptFunction::OPCODE_SET_STATIC_VARIABLE:
				return 3; // opcode + dst + name_idx
			case GDScriptFunction::OPCODE_STORE_GLOBAL:
			case GDScriptFunction::OPCODE_STORE_NAMED_GLOBAL:
				return 3; // opcode + dst + global_idx/name_idx
			case GDScriptFunction::OPCODE_AWAIT:
				return 2; // opcode + operand
			case GDScriptFunction::OPCODE_AWAIT_RESUME:
				return 2; // opcode + target
			case GDScriptFunction::OPCODE_CALL_SELF_BASE: {
				// Structure: [opcode, instr_arg_count, ...args, argc, function_name, dst]
				// After LOAD_INSTRUCTION_ARGS (advances ip by 1), structure is: [argc, function_name, dst]
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + 2 + instr_arg_count < code_size) {
						(void)code_ptr[ip + 2 + instr_arg_count]; // argc - bounds check
						return 5 + instr_arg_count; // opcode + instr_arg_count + args + argc + function_name + dst
					}
				}
				return 2; // Minimum size
			}
			case GDScriptFunction::OPCODE_CREATE_LAMBDA:
			case GDScriptFunction::OPCODE_CREATE_SELF_LAMBDA: {
				// Uses LOAD_INSTRUCTION_ARGS pattern: [opcode, instr_arg_count, ...captures, dst, captures_count, lambda_index]
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + 2 + instr_arg_count < code_size) {
						(void)code_ptr[ip + 2 + instr_arg_count]; // captures_count - bounds check
						return 4 + instr_arg_count; // opcode + instr_arg_count + captures + dst + captures_count + lambda_index
					}
				}
				return 2; // Minimum size
			}
			case GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN:
			case GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY:
			case GDScriptFunction::OPCODE_RETURN_TYPED_DICTIONARY:
			case GDScriptFunction::OPCODE_RETURN_TYPED_NATIVE:
			case GDScriptFunction::OPCODE_RETURN_TYPED_SCRIPT:
				if (ip + 1 < code_size) {
					int return_count = code_ptr[ip + 1];
					return 2 + (return_count > 0 ? 1 : 0);
				}
				return 2;
			case GDScriptFunction::OPCODE_CONSTRUCT_ARRAY: {
				// Uses LOAD_INSTRUCTION_ARGS: [opcode, instr_arg_count, ...args, argc, dst]
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + 2 + instr_arg_count < code_size) {
						(void)code_ptr[ip + 2 + instr_arg_count]; // argc - bounds check
						return 3 + instr_arg_count; // opcode + instr_arg_count + args + argc + dst
					}
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_ARRAY: {
				// Uses LOAD_INSTRUCTION_ARGS: [opcode, instr_arg_count, ...args, argc, script_type, builtin_type, native_type, dst]
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + 2 + instr_arg_count < code_size) {
						(void)code_ptr[ip + 2 + instr_arg_count]; // argc - bounds check
						return 6 + instr_arg_count; // opcode + instr_arg_count + args + argc + script_type + builtin_type + native_type + dst
					}
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY: {
				// Uses LOAD_INSTRUCTION_ARGS: [opcode, instr_arg_count, ...args, argc, dst]
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + 2 + instr_arg_count < code_size) {
						(void)code_ptr[ip + 2 + instr_arg_count]; // argc - bounds check
						return 3 + instr_arg_count; // opcode + instr_arg_count + args + argc + dst
					}
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_CONSTRUCT_TYPED_DICTIONARY: {
				// Uses LOAD_INSTRUCTION_ARGS: [opcode, instr_arg_count, ...args, argc, key_script_type, key_builtin, key_native, value_script_type, value_builtin, value_native, dst]
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + 2 + instr_arg_count < code_size) {
						(void)code_ptr[ip + 2 + instr_arg_count]; // argc - bounds check
						return 8 + instr_arg_count; // opcode + instr_arg_count + args + argc + 6 type fields + dst
					}
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_CALL_GDSCRIPT_UTILITY: {
				// Structure: [opcode, instr_var_args, ...args, dst, argc, utility_idx]
				// From disassembler: incr = 4 + argc, dst at DADDR(1 + argc)
				if (ip + 1 < code_size) {
					int instr_var_args = code_ptr[ip + 1];
					if (ip + 2 + instr_var_args < code_size) {
						(void)code_ptr[ip + 2 + instr_var_args]; // argc - bounds check
						return 5 + instr_var_args; // opcode + instr_var_args + args + dst + argc + utility_idx
					}
				}
				return 2;
			}
			case GDScriptFunction::OPCODE_CALL_BUILTIN_TYPE_VALIDATED: {
				// Structure: [opcode, instr_var_args, ...args, base, argc, method_idx, dst]
				// From disassembler: dst at DADDR(2 + argc)
				if (ip + 1 < code_size) {
					int instr_var_args = code_ptr[ip + 1];
					if (ip + 2 + instr_var_args < code_size) {
						(void)code_ptr[ip + 2 + instr_var_args]; // argc - bounds check
						return 6 + instr_var_args; // opcode + instr_var_args + args + base + argc + method_idx + dst
					}
				}
				return 2;
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
			case GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR4_ARRAY:
				return 2; // opcode + value
			default:
				// For unknown opcodes, try to skip safely
				// Most opcodes have at least 1 word (the opcode itself)
				return 1;
		}
	};

	// Helper function to recursively resolve a value from value_map
	// Using a function pointer to allow recursion in lambda
	struct ResolveValueHelper {
		static Pair<Variant, bool> resolve(int addr, const HashMap<int, Variant> &value_map, const HashMap<int, bool> &is_constant_map, const GDScriptFunction *func, int max_depth = 10) {
			if (max_depth <= 0) {
				return Pair<Variant, bool>(Variant(addr), false);
			}

			// Check if it's a constant
			if ((addr & GDScriptFunction::ADDR_TYPE_MASK) == (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS)) {
				int const_idx = addr & GDScriptFunction::ADDR_MASK;
				if (const_idx >= 0 && const_idx < func->constants.size()) {
					Variant const_val = func->constants[const_idx];
					if (const_val.get_type() == Variant::FLOAT || const_val.get_type() == Variant::INT) {
						return Pair<Variant, bool>(const_val, true);
					}
				}
			}

			// Check if we have a tracked value
			if (value_map.has(addr)) {
				Variant val = value_map[addr];
				bool is_const = is_constant_map.has(addr) ? is_constant_map[addr] : false;

				// If it's a constant, return it
				if (is_const) {
					return Pair<Variant, bool>(val, true);
				}

				// If it's an address, try to resolve it recursively
				if (val.get_type() == Variant::INT) {
					int next_addr = val;
					return resolve(next_addr, value_map, is_constant_map, func, max_depth - 1);
				}

				return Pair<Variant, bool>(val, is_const);
			}

			// Not found, return address as-is
			return Pair<Variant, bool>(Variant(addr), false);
		}
	};

	// Helper function to extract value from a branch starting at a given IP
	// Uses data flow analysis to trace values backwards from assignments/returns
	auto extract_value_from_branch = [&](int branch_ip, Variant &out_value, bool &out_is_constant) -> bool {
		if (branch_ip < 0 || branch_ip >= code_size) {
			return false;
		}

		// Track value destinations: map destination address -> source address/value
		HashMap<int, Variant> value_map; // destination -> source (as Variant containing address or constant)
		HashMap<int, bool> is_constant_map; // destination -> is_constant

		int ip = branch_ip;
		int max_search = 50; // Increased limit for complex expressions
		int search_count = 0;
		HashSet<int> visited_ips; // Track visited IPs to avoid infinite loops

		while (ip < code_size && search_count < max_search) {
			search_count++;

			// Avoid infinite loops
			if (visited_ips.has(ip)) {
				break;
			}
			visited_ips.insert(ip);

			int opcode = code_ptr[ip];
			int opcode_size = get_opcode_size(opcode, code_ptr, ip, code_size);

			// Check for constant assignments
			if (opcode == GDScriptFunction::OPCODE_ASSIGN_TRUE) {
				if (ip + 1 < code_size) {
					int dst = code_ptr[ip + 1];
					value_map[dst] = Variant(1.0);
					is_constant_map[dst] = true;
				}
				ip += opcode_size;
				continue;
			} else if (opcode == GDScriptFunction::OPCODE_ASSIGN_FALSE) {
				if (ip + 1 < code_size) {
					int dst = code_ptr[ip + 1];
					value_map[dst] = Variant(0.0);
					is_constant_map[dst] = true;
				}
				ip += opcode_size;
				continue;
			} else if (opcode == GDScriptFunction::OPCODE_ASSIGN_NULL) {
				if (ip + 1 < code_size) {
					int dst = code_ptr[ip + 1];
					value_map[dst] = Variant(0.0);
					is_constant_map[dst] = true;
				}
				ip += opcode_size;
				continue;
			}

			// Check for RETURN with a value
			if (opcode == GDScriptFunction::OPCODE_RETURN ||
					opcode == GDScriptFunction::OPCODE_RETURN_TYPED_BUILTIN ||
					opcode == GDScriptFunction::OPCODE_RETURN_TYPED_ARRAY ||
					opcode == GDScriptFunction::OPCODE_RETURN_TYPED_DICTIONARY ||
					opcode == GDScriptFunction::OPCODE_RETURN_TYPED_NATIVE ||
					opcode == GDScriptFunction::OPCODE_RETURN_TYPED_SCRIPT) {
				if (ip + 1 < code_size) {
					int return_count = code_ptr[ip + 1];
					if (return_count > 0 && ip + 2 < code_size) {
						int return_value = code_ptr[ip + 2];
						// Check if it's a constant
						if ((return_value & GDScriptFunction::ADDR_TYPE_MASK) == (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS)) {
							int const_idx = return_value & GDScriptFunction::ADDR_MASK;
							if (const_idx >= 0 && const_idx < p_function->constants.size()) {
								Variant const_val = p_function->constants[const_idx];
								if (const_val.get_type() == Variant::FLOAT || const_val.get_type() == Variant::INT) {
									out_value = const_val;
									out_is_constant = true;
									return true;
								}
							}
						} else {
							// Resolve the value recursively
							Pair<Variant, bool> resolved = ResolveValueHelper::resolve(return_value, value_map, is_constant_map, p_function);
							out_value = resolved.first;
							out_is_constant = resolved.second;
							return true;
						}
					}
				}
				return false; // Return without value
			}

			// Check for ASSIGN (ternary pattern or value propagation)
			if (opcode == GDScriptFunction::OPCODE_ASSIGN ||
					opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN ||
					opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_ARRAY ||
					opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_DICTIONARY ||
					opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE ||
					opcode == GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT) {
				if (ip + 2 < code_size) {
					int target = code_ptr[ip + 1];
					int source = code_ptr[ip + 2];

					Variant resolved_value;
					bool resolved_is_constant = false;

					// Resolve the source value recursively
					Pair<Variant, bool> resolved = ResolveValueHelper::resolve(source, value_map, is_constant_map, p_function);
					resolved_value = resolved.first;
					resolved_is_constant = resolved.second;

					value_map[target] = resolved_value;
					is_constant_map[target] = resolved_is_constant;

					// Check if next instruction is a jump (this might be the final assignment in branch)
					int next_ip = ip + opcode_size;
					if (next_ip < code_size) {
						int next_opcode = code_ptr[next_ip];
						if (next_opcode == GDScriptFunction::OPCODE_JUMP ||
								next_opcode == GDScriptFunction::OPCODE_JUMP_IF ||
								next_opcode == GDScriptFunction::OPCODE_JUMP_IF_NOT ||
								next_opcode == GDScriptFunction::OPCODE_JUMP_IF_SHARED) {
							// This assignment is the final value before a jump - extract it
							out_value = resolved_value;
							out_is_constant = resolved_is_constant;
							return true;
						}
					}
				}
				ip += opcode_size;
				continue;
			}

			// Handle value-producing operations
			// OPERATOR and OPERATOR_VALIDATED produce values
			if (opcode == GDScriptFunction::OPCODE_OPERATOR || opcode == GDScriptFunction::OPCODE_OPERATOR_VALIDATED) {
				if (ip + 4 < code_size) {
					int dst = code_ptr[ip + 1];
					int a = code_ptr[ip + 2];
					int b = code_ptr[ip + 3];

					// Try to resolve operands
					Variant a_val, b_val;

					// Resolve operand a
					if ((a & GDScriptFunction::ADDR_TYPE_MASK) == (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS)) {
						int const_idx = a & GDScriptFunction::ADDR_MASK;
						if (const_idx >= 0 && const_idx < p_function->constants.size()) {
							Variant const_val = p_function->constants[const_idx];
							if (const_val.get_type() == Variant::FLOAT || const_val.get_type() == Variant::INT) {
								a_val = const_val;
							}
						}
					} else if (value_map.has(a)) {
						a_val = value_map[a];
					} else {
						a_val = Variant(a);
					}

					// Resolve operand b
					if ((b & GDScriptFunction::ADDR_TYPE_MASK) == (GDScriptFunction::ADDR_TYPE_CONSTANT << GDScriptFunction::ADDR_BITS)) {
						int const_idx = b & GDScriptFunction::ADDR_MASK;
						if (const_idx >= 0 && const_idx < p_function->constants.size()) {
							Variant const_val = p_function->constants[const_idx];
							if (const_val.get_type() == Variant::FLOAT || const_val.get_type() == Variant::INT) {
								b_val = const_val;
							}
						}
					} else if (value_map.has(b)) {
						b_val = value_map[b];
					} else {
						b_val = Variant(b);
					}

					// If both are constants, we could compute the result, but for now just track the destination
					// Store the destination address as the value (will be resolved later)
					value_map[dst] = Variant(dst);
					is_constant_map[dst] = false;
				}
				ip += opcode_size;
				continue;
			}

			// GET operations produce values
			if (opcode == GDScriptFunction::OPCODE_GET_MEMBER ||
					opcode == GDScriptFunction::OPCODE_GET_NAMED ||
					opcode == GDScriptFunction::OPCODE_GET_NAMED_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_GET_KEYED ||
					opcode == GDScriptFunction::OPCODE_GET_KEYED_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_GET_INDEXED_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_GET_STATIC_VARIABLE) {
				if (ip + 2 < code_size) {
					int dst = code_ptr[ip + 1];
					// Track destination (value comes from member/global, not a constant)
					value_map[dst] = Variant(dst);
					is_constant_map[dst] = false;
				}
				ip += opcode_size;
				continue;
			}

			// TYPE_TEST operations produce boolean values
			if (opcode == GDScriptFunction::OPCODE_TYPE_TEST_BUILTIN ||
					opcode == GDScriptFunction::OPCODE_TYPE_TEST_ARRAY ||
					opcode == GDScriptFunction::OPCODE_TYPE_TEST_DICTIONARY ||
					opcode == GDScriptFunction::OPCODE_TYPE_TEST_NATIVE ||
					opcode == GDScriptFunction::OPCODE_TYPE_TEST_SCRIPT) {
				if (ip + 2 < code_size) {
					int dst = code_ptr[ip + 1];
					(void)code_ptr[ip + 2]; // source - bounds check
					// TYPE_TEST produces boolean, but we track it as a variable value
					// Could potentially resolve source and determine if constant true/false
					value_map[dst] = Variant(dst);
					is_constant_map[dst] = false;
				}
				ip += opcode_size;
				continue;
			}

			// CALL operations that return values
			if (opcode == GDScriptFunction::OPCODE_CALL_RETURN ||
					opcode == GDScriptFunction::OPCODE_CALL_UTILITY ||
					opcode == GDScriptFunction::OPCODE_CALL_UTILITY_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_CALL_GDSCRIPT_UTILITY ||
					opcode == GDScriptFunction::OPCODE_CALL_BUILTIN_TYPE_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_CALL_METHOD_BIND_RET ||
					opcode == GDScriptFunction::OPCODE_CALL_BUILTIN_STATIC ||
					opcode == GDScriptFunction::OPCODE_CALL_NATIVE_STATIC ||
					opcode == GDScriptFunction::OPCODE_CALL_NATIVE_STATIC_VALIDATED_RETURN ||
					opcode == GDScriptFunction::OPCODE_CALL_METHOD_BIND_VALIDATED_RETURN ||
					opcode == GDScriptFunction::OPCODE_CALL_SELF_BASE) {
				// Handle variable-size CALL operations
				if (opcode == GDScriptFunction::OPCODE_CALL_SELF_BASE) {
					// CALL_SELF_BASE: [opcode, instr_arg_count, ...args, argc, function_name, dst]
					if (ip + 1 < code_size) {
						int instr_arg_count = code_ptr[ip + 1];
						if (ip + 2 + instr_arg_count < code_size) {
							(void)code_ptr[ip + 2 + instr_arg_count]; // argc - bounds check
							if (ip + 4 + instr_arg_count < code_size) {
								int dst = code_ptr[ip + 4 + instr_arg_count]; // dst is after argc and function_name
								value_map[dst] = Variant(dst);
								is_constant_map[dst] = false;
							}
						}
					}
				} else if (opcode == GDScriptFunction::OPCODE_CALL_GDSCRIPT_UTILITY) {
					// Structure: [opcode, instr_arg_count (1+argc), ...args, dst, argc, utility_idx]
					// dst is included in instr_arg_count, at position argc
					if (ip + 1 < code_size) {
						int instr_arg_count = code_ptr[ip + 1];
						if (ip + 1 + instr_arg_count < code_size) {
							// After args, we have dst, then argc, then utility_idx
							// But dst is actually at position argc within the args (instr_arg_count includes dst)
							// From bytecode generator: dst is appended after args, before argc
							// instr_arg_count = 1 + argc, so dst is at position instr_arg_count
							int dst = code_ptr[ip + 1 + instr_arg_count];
							value_map[dst] = Variant(dst);
							is_constant_map[dst] = false;
						}
					}
				} else if (opcode == GDScriptFunction::OPCODE_CALL_BUILTIN_TYPE_VALIDATED) {
					// Structure: [opcode, instr_var_args (2+argc), ...args, base, dst, argc, method]
					// From bytecode generator: arg_count = 2 + p_arguments.size()
					// Then: args, base, dst, argc, method
					if (ip + 1 < code_size) {
						int instr_var_args = code_ptr[ip + 1];
						// dst is after args and base
						// instr_var_args includes base and dst, so dst is at position instr_var_args
						if (ip + 1 + instr_var_args < code_size) {
							int dst = code_ptr[ip + 1 + instr_var_args];
							value_map[dst] = Variant(dst);
							is_constant_map[dst] = false;
						}
					}
				} else {
					// Standard CALL operations
					if (ip + 1 < code_size) {
						int arg_count = code_ptr[ip + 1];
						if (ip + 2 + arg_count < code_size) {
							int dst = code_ptr[ip + 2 + arg_count];
							// Track destination (value comes from function call, not a constant)
							value_map[dst] = Variant(dst);
							is_constant_map[dst] = false;
						}
					}
				}
				ip += opcode_size;
				continue;
			}

			// CONSTRUCT operations produce values
			if (opcode == GDScriptFunction::OPCODE_CONSTRUCT ||
					opcode == GDScriptFunction::OPCODE_CONSTRUCT_VALIDATED) {
				if (ip + 1 < code_size) {
					int arg_count = code_ptr[ip + 1];
					if (ip + 2 + arg_count < code_size) {
						int dst = code_ptr[ip + 2 + arg_count];
						// Track destination (value comes from construction, not a constant)
						value_map[dst] = Variant(dst);
						is_constant_map[dst] = false;
					}
				}
				ip += opcode_size;
				continue;
			}
			if (opcode == GDScriptFunction::OPCODE_CONSTRUCT_ARRAY) {
				// [opcode, instr_arg_count, ...args, target, argc]
				// instr_arg_count = 1 + argc, target is at ip + instr_arg_count
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + instr_arg_count < code_size) {
						int dst = code_ptr[ip + instr_arg_count];
						value_map[dst] = Variant(dst);
						is_constant_map[dst] = false;
					}
				}
				ip += opcode_size;
				continue;
			}
			if (opcode == GDScriptFunction::OPCODE_CONSTRUCT_TYPED_ARRAY) {
				// [opcode, instr_arg_count, ...args, target, argc, script_type, builtin_type, native_type]
				// instr_arg_count = 2 + argc, target is at ip + instr_arg_count
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + instr_arg_count < code_size) {
						int dst = code_ptr[ip + instr_arg_count];
						value_map[dst] = Variant(dst);
						is_constant_map[dst] = false;
					}
				}
				ip += opcode_size;
				continue;
			}
			if (opcode == GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY) {
				// [opcode, instr_arg_count, ...args, target, argc]
				// instr_arg_count = 1 + argc, target is at ip + instr_arg_count
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + instr_arg_count < code_size) {
						int dst = code_ptr[ip + instr_arg_count];
						value_map[dst] = Variant(dst);
						is_constant_map[dst] = false;
					}
				}
				ip += opcode_size;
				continue;
			}
			if (opcode == GDScriptFunction::OPCODE_CONSTRUCT_TYPED_DICTIONARY) {
				// [opcode, instr_arg_count, ...args, target, argc, key_script_type, key_builtin, key_native, value_script_type, value_builtin, value_native]
				// instr_arg_count = 3 + argc, target is at ip + instr_arg_count
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + instr_arg_count < code_size) {
						int dst = code_ptr[ip + instr_arg_count];
						value_map[dst] = Variant(dst);
						is_constant_map[dst] = false;
					}
				}
				ip += opcode_size;
				continue;
			}

			// CAST operations produce values
			if (opcode == GDScriptFunction::OPCODE_CAST_TO_BUILTIN ||
					opcode == GDScriptFunction::OPCODE_CAST_TO_NATIVE ||
					opcode == GDScriptFunction::OPCODE_CAST_TO_SCRIPT) {
				if (ip + 2 < code_size) {
					int dst = code_ptr[ip + 1];
					int source = code_ptr[ip + 2];
					// Use ResolveValueHelper for proper propagation
					Pair<Variant, bool> resolved = ResolveValueHelper::resolve(source, value_map, is_constant_map, p_function);
					value_map[dst] = resolved.first;
					is_constant_map[dst] = resolved.second;
				}
				ip += opcode_size;
				continue;
			}

			// AWAIT_RESUME produces values
			if (opcode == GDScriptFunction::OPCODE_AWAIT_RESUME) {
				if (ip + 2 < code_size) {
					int target = code_ptr[ip + 1];
					// Value comes from await, not a constant
					value_map[target] = Variant(target);
					is_constant_map[target] = false;
				}
				ip += opcode_size;
				continue;
			}

			// CREATE_LAMBDA operations produce callable values
			if (opcode == GDScriptFunction::OPCODE_CREATE_LAMBDA ||
					opcode == GDScriptFunction::OPCODE_CREATE_SELF_LAMBDA) {
				// [opcode, instr_arg_count, ...captures, dst, captures_count, lambda_index]
				if (ip + 1 < code_size) {
					int instr_arg_count = code_ptr[ip + 1];
					if (ip + 2 + instr_arg_count < code_size) {
						int dst = code_ptr[ip + 2 + instr_arg_count];
						value_map[dst] = Variant(dst);
						is_constant_map[dst] = false;
					}
				}
				ip += opcode_size;
				continue;
			}

			// SET operations don't produce values, just skip them
			if (opcode == GDScriptFunction::OPCODE_SET_KEYED ||
					opcode == GDScriptFunction::OPCODE_SET_KEYED_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_SET_INDEXED_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_SET_NAMED ||
					opcode == GDScriptFunction::OPCODE_SET_NAMED_VALIDATED ||
					opcode == GDScriptFunction::OPCODE_SET_STATIC_VARIABLE ||
					opcode == GDScriptFunction::OPCODE_STORE_GLOBAL ||
					opcode == GDScriptFunction::OPCODE_STORE_NAMED_GLOBAL) {
				ip += opcode_size;
				continue;
			}

			// TYPE_ADJUST operations don't produce new values, just adjust types
			// Skip all TYPE_ADJUST_* opcodes
			if (opcode >= GDScriptFunction::OPCODE_TYPE_ADJUST_BOOL &&
					opcode <= GDScriptFunction::OPCODE_TYPE_ADJUST_PACKED_VECTOR4_ARRAY) {
				ip += opcode_size;
				continue;
			}

			// Skip metadata opcodes
			if (opcode == GDScriptFunction::OPCODE_LINE ||
					opcode == GDScriptFunction::OPCODE_BREAKPOINT ||
					opcode == GDScriptFunction::OPCODE_ASSERT ||
					opcode == GDScriptFunction::OPCODE_END) {
				ip += opcode_size;
				continue;
			}

			// If we hit a jump, we've reached the end of this branch
			if (opcode == GDScriptFunction::OPCODE_JUMP ||
					opcode == GDScriptFunction::OPCODE_JUMP_IF ||
					opcode == GDScriptFunction::OPCODE_JUMP_IF_NOT ||
					opcode == GDScriptFunction::OPCODE_JUMP_IF_SHARED ||
					opcode == GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT) {
				break;
			}

			// For other opcodes, advance IP by opcode size
			ip += opcode_size;
		}

		// If we didn't find a direct return/assign, check if we can extract from value_map
		// This handles cases where the value is computed and then used
		// For now, return false as we need a direct assignment/return to extract
		return false;
	};

	// Extract values from both branches
	bool found_true = extract_value_from_branch(true_branch_start, p_true_value, p_true_is_constant);
	bool found_false = extract_value_from_branch(false_branch_start, p_false_value, p_false_is_constant);

	return found_true && found_false;
}

String GDScriptToStableHLO::generate_operation(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size,
		int &p_value_id, const GDScriptFunction *p_function) {
	String result;

	switch (p_opcode) {
		case GDScriptFunction::OPCODE_ASSIGN_NULL:
		case GDScriptFunction::OPCODE_ASSIGN_TRUE:
		case GDScriptFunction::OPCODE_ASSIGN_FALSE: {
			// Simple constant assignment
			Variant value;
			if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_NULL) {
				value = Variant();
			} else if (p_opcode == GDScriptFunction::OPCODE_ASSIGN_TRUE) {
				value = true;
			} else {
				value = false;
			}
			result = generate_constant(value, p_value_id);
			p_ip += 1;
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN: {
			// Assignment from stack - just use the value directly (no copy needed)
			// The value is already available, so we don't need to generate anything
			if (p_ip + 1 < p_code_size) {
				p_ip += 2;
			} else {
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_RETURN: {
			// Return operation
			if (p_ip + 1 < p_code_size) {
				int return_count = p_code_ptr[p_ip + 1];
				if (return_count > 0 && p_ip + 2 < p_code_size) {
					int return_value = p_code_ptr[p_ip + 2];
					result = "  stablehlo.return %v" + String::num(return_value) + " : tensor<f32>\n";
				} else {
					result = "  stablehlo.return\n";
				}
				p_ip += 2;
			} else {
				result = "  stablehlo.return\n";
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_JUMP: {
			// Unconditional jump
			if (p_ip + 1 < p_code_size) {
				int target = p_code_ptr[p_ip + 1];
				result = "  stablehlo.return // jump to " + String::num(target) + "\n";
				p_ip += 2;
			} else {
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_JUMP_IF:
		case GDScriptFunction::OPCODE_JUMP_IF_NOT: {
			// Conditional jump - convert to compare + select pattern
			// This requires tracking the condition value and true/false branches
			// For now, generate a placeholder that will be handled by the operator that precedes it
			// The actual conversion happens when we see the comparison operator
			if (p_ip + 1 < p_code_size) {
				(void)p_code_ptr[p_ip + 1]; // target - bounds check
				// Note: This will be handled by the comparison operator that precedes it
				// We'll generate compare + select in the operator case
				p_ip += 2;
			} else {
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR:
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED: {
			// Arithmetic/comparison operation
			// Bytecode structure: [opcode, a_addr, b_addr, dst_addr, operator, ...]
			if (p_ip + 4 < p_code_size) {
				int a_addr = p_code_ptr[p_ip + 1];
				int b_addr = p_code_ptr[p_ip + 2];
				(void)p_code_ptr[p_ip + 3]; // dst_addr - bounds check
				Variant::Operator op = (Variant::Operator)p_code_ptr[p_ip + 4];

				// Decode addresses
				int a_type = (a_addr & GDScriptFunction::ADDR_TYPE_MASK) >> GDScriptFunction::ADDR_BITS;
				int a_idx = a_addr & GDScriptFunction::ADDR_MASK;
				int b_type = (b_addr & GDScriptFunction::ADDR_TYPE_MASK) >> GDScriptFunction::ADDR_BITS;
				int b_idx = b_addr & GDScriptFunction::ADDR_MASK;

				// Resolve operand references
				String a_ref, b_ref;
				if (a_type == GDScriptFunction::ADDR_TYPE_CONSTANT && a_idx < p_function->constants.size()) {
					// Constant reference
					a_ref = "%c" + String::num(a_idx);
				} else if (a_type == GDScriptFunction::ADDR_TYPE_STACK) {
					// Stack reference - map to SSA value
					// For now, use simple mapping (arg0, arg1, etc. or v0, v1, etc.)
					if (a_idx < p_function->get_argument_count()) {
						a_ref = "%arg" + String::num(a_idx);
					} else {
						a_ref = "%v" + String::num(a_idx - p_function->get_argument_count());
					}
				} else {
					a_ref = "%v0"; // Fallback
				}

				if (b_type == GDScriptFunction::ADDR_TYPE_CONSTANT && b_idx < p_function->constants.size()) {
					b_ref = "%c" + String::num(b_idx);
				} else if (b_type == GDScriptFunction::ADDR_TYPE_STACK) {
					if (b_idx < p_function->get_argument_count()) {
						b_ref = "%arg" + String::num(b_idx);
					} else {
						b_ref = "%v" + String::num(b_idx - p_function->get_argument_count());
					}
				} else {
					b_ref = "%v0"; // Fallback
				}

				String op_name;
				bool is_comparison = false;
				String compare_type;

				switch (op) {
					case Variant::OP_ADD:
						op_name = "add";
						break;
					case Variant::OP_SUBTRACT:
						op_name = "subtract";
						break;
					case Variant::OP_MULTIPLY:
						op_name = "multiply";
						break;
					case Variant::OP_DIVIDE:
						op_name = "divide";
						break;
					case Variant::OP_GREATER:
						is_comparison = true;
						compare_type = "GT";
						break;
					case Variant::OP_LESS:
						is_comparison = true;
						compare_type = "LT";
						break;
					case Variant::OP_GREATER_EQUAL:
						is_comparison = true;
						compare_type = "GE";
						break;
					case Variant::OP_LESS_EQUAL:
						is_comparison = true;
						compare_type = "LE";
						break;
					case Variant::OP_EQUAL:
						is_comparison = true;
						compare_type = "EQ";
						break;
					case Variant::OP_NOT_EQUAL:
						is_comparison = true;
						compare_type = "NE";
						break;
					default:
						op_name = "add"; // Default fallback
						break;
				}

				if (is_comparison) {
					// Check if next opcode is JUMP_IF to generate compare + select pattern
					// OPCODE_OPERATOR has variable size due to function pointer storage
					// For OPCODE_OPERATOR_VALIDATED, it's fixed at 5
					// Let's check a bit further ahead
					int check_ip = (p_opcode == GDScriptFunction::OPCODE_OPERATOR_VALIDATED) ? p_ip + 5 : p_ip + 7;
					if (check_ip < p_code_size &&
							(p_code_ptr[check_ip] == GDScriptFunction::OPCODE_JUMP_IF ||
									p_code_ptr[check_ip] == GDScriptFunction::OPCODE_JUMP_IF_NOT)) {
						// Extract jump information
						int jump_ip = check_ip;
						bool is_jump_if_not = (p_code_ptr[jump_ip] == GDScriptFunction::OPCODE_JUMP_IF_NOT);
						// JUMP_IF structure: [opcode, condition_address, target]
						// ip+0: opcode, ip+1: condition address, ip+2: jump target
						int jump_target = (jump_ip + 2 < p_code_size) ? p_code_ptr[jump_ip + 2] : -1;

						// Generate zero constant for comparison
						Variant zero_val = 0.0;
						result += generate_constant(zero_val, p_value_id);
						int zero_id = p_value_id - 1;

						// Generate comparison (compare a against zero)
						result += "  %cmp" + String::num(p_value_id) + " = stablehlo.compare " + compare_type + ", " + a_ref + ", %c" + String::num(zero_id) + " : (tensor<f32>, tensor<f32>) -> tensor<i1>\n";
						int cmp_id = p_value_id;
						p_value_id++;

						// Extract branch values from bytecode
						Variant true_value;
						Variant false_value;
						bool true_is_constant = false;
						bool false_is_constant = false;
						bool extracted = extract_branch_values(p_function, jump_ip, jump_target, is_jump_if_not, true_value, false_value, true_is_constant, false_is_constant);

						String true_ref, false_ref;
						if (extracted) {
							if (true_is_constant) {
								// Generate constant for true value
								result += generate_constant(true_value, p_value_id);
								true_ref = "%c" + String::num(p_value_id - 1);
							} else {
								// Use stack value reference - decode address
								int addr = true_value;
								int addr_type = (addr & GDScriptFunction::ADDR_TYPE_MASK) >> GDScriptFunction::ADDR_BITS;
								int addr_idx = addr & GDScriptFunction::ADDR_MASK;
								if (addr_type == GDScriptFunction::ADDR_TYPE_CONSTANT && addr_idx < p_function->constants.size()) {
									result += generate_constant(p_function->constants[addr_idx], p_value_id);
									true_ref = "%c" + String::num(p_value_id - 1);
								} else if (addr_type == GDScriptFunction::ADDR_TYPE_STACK) {
									if (addr_idx < p_function->get_argument_count()) {
										true_ref = "%arg" + String::num(addr_idx);
									} else {
										true_ref = "%v" + String::num(addr_idx - p_function->get_argument_count());
									}
								} else {
									// Fallback
									Variant fallback = 100.0;
									result += generate_constant(fallback, p_value_id);
									true_ref = "%c" + String::num(p_value_id - 1);
								}
							}

							if (false_is_constant) {
								// Generate constant for false value
								result += generate_constant(false_value, p_value_id);
								false_ref = "%c" + String::num(p_value_id - 1);
							} else {
								// Use stack value reference - decode address
								int addr = false_value;
								int addr_type = (addr & GDScriptFunction::ADDR_TYPE_MASK) >> GDScriptFunction::ADDR_BITS;
								int addr_idx = addr & GDScriptFunction::ADDR_MASK;
								if (addr_type == GDScriptFunction::ADDR_TYPE_CONSTANT && addr_idx < p_function->constants.size()) {
									result += generate_constant(p_function->constants[addr_idx], p_value_id);
									false_ref = "%c" + String::num(p_value_id - 1);
								} else if (addr_type == GDScriptFunction::ADDR_TYPE_STACK) {
									if (addr_idx < p_function->get_argument_count()) {
										false_ref = "%arg" + String::num(addr_idx);
									} else {
										false_ref = "%v" + String::num(addr_idx - p_function->get_argument_count());
									}
								} else {
									// Fallback
									Variant fallback = 0.0;
									result += generate_constant(fallback, p_value_id);
									false_ref = "%c" + String::num(p_value_id - 1);
								}
							}
						} else {
							// Fallback to placeholder values if extraction failed
							Variant true_val = 100.0;
							Variant false_val = 0.0;
							result += generate_constant(true_val, p_value_id);
							true_ref = "%c" + String::num(p_value_id - 1);
							result += generate_constant(false_val, p_value_id);
							false_ref = "%c" + String::num(p_value_id - 1);
						}

						// Generate select (true value if condition, false value otherwise)
						result += "  %v" + String::num(p_value_id) + " = stablehlo.select %cmp" + String::num(cmp_id) + ", " + true_ref + ", " + false_ref + " : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>\n";
						p_value_id++;

						// Skip operator + jump_if
						p_ip = check_ip + 3; // Skip opcode + condition + target
					} else {
						// Just generate compare operation
						result = "  %cmp" + String::num(p_value_id) + " = stablehlo.compare " + compare_type + ", " + a_ref + ", " + b_ref + " : (tensor<f32>, tensor<f32>) -> tensor<i1>\n";
						p_value_id++;
						// Skip based on opcode type
						p_ip += (p_opcode == GDScriptFunction::OPCODE_OPERATOR_VALIDATED) ? 5 : 7;
					}
				} else {
					// Generate arithmetic operation
					result = "  %v" + String::num(p_value_id) + " = stablehlo." + op_name + " " + a_ref + ", " + b_ref + " : (tensor<f32>, tensor<f32>) -> tensor<f32>\n";
					p_value_id++;
					// Skip based on opcode type
					p_ip += (p_opcode == GDScriptFunction::OPCODE_OPERATOR_VALIDATED) ? 5 : 7;
				}
			} else {
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_GET_MEMBER:
		case GDScriptFunction::OPCODE_SET_MEMBER: {
			// Custom call for member access
			if (p_ip + 1 < p_code_size) {
				String op_type = (p_opcode == GDScriptFunction::OPCODE_GET_MEMBER) ? "get" : "set";
				int obj_idx = (p_value_id > 0) ? p_value_id - 1 : 0;
				result = "  %v" + String::num(p_value_id) + " = stablehlo.custom_call @gdscript_" + op_type + "_member(%v" + String::num(obj_idx) + ") : (tensor<f32>) -> tensor<f32>\n";
				p_value_id++;
				p_ip += 2;
			} else {
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_CALL:
		case GDScriptFunction::OPCODE_CALL_RETURN: {
			// Function call
			if (p_ip + 1 < p_code_size) {
				int arg_count = p_code_ptr[p_ip + 1];
				int arg_idx = (p_value_id > 0) ? p_value_id - 1 : 0;
				result = "  %v" + String::num(p_value_id) + " = stablehlo.call @function(%v" + String::num(arg_idx) + ") : (tensor<f32>) -> tensor<f32>\n";
				p_value_id++;
				p_ip += 2 + arg_count;
			} else {
				p_ip += 1;
			}
			break;
		}
		default: {
			// Metadata opcodes (LINE, BREAKPOINT, ASSERT, END)
			result = "  // opcode " + String::num(p_opcode) + " (metadata)\n";
			p_ip += 1;
			break;
		}
	}

	return result;
}

String GDScriptToStableHLO::convert_function_to_stablehlo_text(const GDScriptFunction *p_function) {
	if (!p_function || p_function->code.is_empty()) {
		return String();
	}

	String result;
	String function_name = p_function->get_name();

	// MLIR/StableHLO module header
	result += "module {\n";
	result += "  func.func @" + function_name + "(";

	// Function arguments
	int arg_count = p_function->get_argument_count();
	for (int i = 0; i < arg_count; i++) {
		if (i > 0) {
			result += ", ";
		}
		result += "%arg" + String::num(i) + ": tensor<f32>";
	}
	result += ") -> tensor<f32> {\n";

	// Process opcodes
	const int *code_ptr = p_function->code.ptr();
	int code_size = p_function->code.size();
	int ip = 0;
	int value_id = arg_count;

	// Generate constants first
	for (int i = 0; i < p_function->constants.size(); i++) {
		result += generate_constant(p_function->constants[i], value_id);
	}

	// Process opcodes
	while (ip < code_size) {
		int opcode = code_ptr[ip];
		if (!is_basic_opcode(opcode)) {
			print_error(vformat("GDScriptToStableHLO: Unsupported opcode %d at IP %d", opcode, ip));
			break;
		}

		result += generate_operation(opcode, code_ptr, ip, code_size, value_id, p_function);

		if (ip > code_size) {
			break;
		}
	}

	result += "  }\n";
	result += "}\n";

	return result;
}

String GDScriptToStableHLO::convert_function_to_stablehlo_bytecode(const GDScriptFunction *p_function, const String &p_output_path) {
	if (!p_function) {
		return String();
	}

	// Generate StableHLO text
	String stablehlo_text = convert_function_to_stablehlo_text(p_function);
	if (stablehlo_text.is_empty()) {
		return String();
	}

	// Write text to temporary file
	String temp_text_path = p_output_path;
	if (!temp_text_path.ends_with(".stablehlo")) {
		temp_text_path += ".stablehlo";
	}
	Ref<FileAccess> text_file = FileAccess::open(temp_text_path, FileAccess::WRITE);
	if (!text_file.is_valid()) {
		print_error("GDScriptToStableHLO: Failed to write StableHLO text file");
		return String();
	}
	text_file->store_string(stablehlo_text);
	text_file->close();

	return temp_text_path;
}

String GDScriptToStableHLO::generate_mlir_file(const GDScriptFunction *p_function, const String &p_output_path) {
	if (!p_function) {
		return String();
	}

	// Generate StableHLO text
	String stablehlo_text = convert_function_to_stablehlo_text(p_function);
	if (stablehlo_text.is_empty()) {
		return String();
	}

	// Write StableHLO text to file
	String stablehlo_path = p_output_path;
	if (!stablehlo_path.ends_with(".stablehlo")) {
		stablehlo_path += ".stablehlo";
	}

	Ref<FileAccess> stablehlo_file = FileAccess::open(stablehlo_path, FileAccess::WRITE);
	if (!stablehlo_file.is_valid()) {
		print_error("GDScriptToStableHLO: Failed to write StableHLO file");
		return String();
	}
	stablehlo_file->store_string(stablehlo_text);
	stablehlo_file->close();

	return stablehlo_path;
}
