/**************************************************************************/
/*  ir_interpreter.h                                                      */
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

#pragma once
#include "ir.h"
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

namespace gdscript {

// Simple IR interpreter for testing without needing full RISC-V execution
class IRInterpreter {
public:
	using Value = std::variant<int64_t, double, std::string, bool>;

	IRInterpreter(const IRProgram &program);

	// Execute a function and return result
	Value call(const std::string &function_name, const std::vector<Value> &args = {});

	// Get last error
	std::string get_error() const { return m_error; }

private:
	struct ExecutionContext {
		std::unordered_map<int, Value> registers; // Virtual register -> value
		std::unordered_map<std::string, size_t> labels; // Label -> instruction index
		size_t pc = 0; // Program counter
		bool returned = false;
		Value return_value;
	};

	void execute_function(const IRFunction &func, ExecutionContext &ctx);
	void execute_instruction(const IRInstruction &instr, ExecutionContext &ctx);
	Value get_register(ExecutionContext &ctx, int reg);

	// Helper functions
	int64_t get_int(const Value &v) const;
	double get_double(const Value &v) const;
	bool get_bool(const Value &v) const;
	std::string get_string(const Value &v) const;

	Value binary_op(const Value &left, const Value &right, IROpcode op);
	Value unary_op(const Value &operand, IROpcode op);
	Value compare_op(const Value &left, const Value &right, IROpcode op);

	const IRProgram &m_program;
	std::unordered_map<std::string, const IRFunction *> m_function_map;
	std::string m_error;
};

} // namespace gdscript
