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
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFLICTING. */
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
		// Advance IP by at least 1 (opcode)
		ip += 1;
		if (ip >= code_size) {
			break;
		}
	}

	return true;
}

String GDScriptToStableHLO::generate_constant(const Variant &p_value, int &p_value_id) {
	String value_str = p_value;
	String result = "  %c" + String::num(p_value_id) + " = stablehlo.constant dense<" + value_str + "> : tensor<f32>\n";
	p_value_id++;
	return result;
}

String GDScriptToStableHLO::generate_operation(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size,
                                                int &p_value_id) {
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
			// Assignment from stack
			if (p_ip + 1 < p_code_size) {
				int src = p_code_ptr[p_ip + 1];
				result = "  %v" + String::num(p_value_id) + " = stablehlo.copy %v" + String::num(src) + " : tensor<f32>\n";
				p_value_id++;
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
			// Conditional jump
			if (p_ip + 1 < p_code_size) {
				int target = p_code_ptr[p_ip + 1];
				int cond_idx = (p_value_id > 0) ? p_value_id - 1 : 0;
				result = "  stablehlo.if %v" + String::num(cond_idx) + " -> (tensor<f32>) { stablehlo.return // jump to " + String::num(target) + " } : (tensor<f32>) { }\n";
				p_ip += 2;
			} else {
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR:
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED: {
			// Arithmetic operation
			if (p_ip + 3 < p_code_size) {
				int op = p_code_ptr[p_ip + 1];
				int a = p_code_ptr[p_ip + 2];
				int b = p_code_ptr[p_ip + 3];
				String op_name = "add";
				result = "  %v" + String::num(p_value_id) + " = stablehlo." + op_name + " %v" + String::num(a) + ", %v" + String::num(b) + " : (tensor<f32>, tensor<f32>) -> tensor<f32>\n";
				p_value_id++;
				p_ip += 4;
			} else {
				p_ip += 1;
			}
			break;
		}
		case GDScriptFunction::OPCODE_GET_MEMBER:
		case GDScriptFunction::OPCODE_SET_MEMBER: {
			// Custom call for member access
			if (p_ip + 1 < p_code_size) {
				int name_idx = p_code_ptr[p_ip + 1];
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
	if (!p_function || !p_function->_code_ptr || p_function->_code_size == 0) {
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
	const int *code_ptr = p_function->_code_ptr;
	int code_size = p_function->_code_size;
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
		
		result += generate_operation(opcode, code_ptr, ip, code_size, value_id);
		
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

