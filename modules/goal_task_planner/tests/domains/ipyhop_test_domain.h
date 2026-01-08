/**************************************************************************/
/*  ipyhop_test_domain.h                                                  */
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

#include "../../src/domain.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

// IPyHOP test domain - used for compatibility testing

// Wrapper class used to construct Callables for free functions
class IPyHOPTestDomainCallable {
public:
	static Variant action_transfer_flag(Dictionary p_state, int p_flag_key_1, int p_flag_key_2);
	static Dictionary action_putv(Dictionary p_state, int p_flag_val);
	static Variant action_getv(Dictionary p_state, int p_flag_val);
	static Variant task_method_1_1(Dictionary p_state);
	static Variant task_method_1_2(Dictionary p_state);
	static Variant task_method_1_3(Dictionary p_state);
	static Variant task_method_2_1(Dictionary p_state);
	static Variant task_method_2_2(Dictionary p_state);
	static Variant task_method_m_err(Dictionary p_state);
	static Variant task_method_m0(Dictionary p_state);
	static Variant task_method_m1(Dictionary p_state);
	static Variant task_method_m_need0(Dictionary p_state);
	static Variant task_method_m_need1(Dictionary p_state);
};

namespace IPyHOPTestDomain {

// Action: transfer_flag(state, flag_key_1, flag_key_2) - sets flag[flag_key_2] = True if flag[flag_key_1] is True
Variant action_transfer_flag(Dictionary state, int flag_key_1, int flag_key_2) {
	Dictionary flag_dict;
	if (state.has("flag")) {
		flag_dict = state["flag"];
	} else {
		flag_dict = Dictionary();
	}

	// Check if flag[flag_key_1] is True
	if (flag_dict.has(flag_key_1) && bool(flag_dict[flag_key_1])) {
		Dictionary new_state = state.duplicate();
		flag_dict = flag_dict.duplicate();
		flag_dict[flag_key_2] = true;
		new_state["flag"] = flag_dict;
		return new_state;
	}
	// Precondition not met, return false to indicate failure
	return false;
}

// Action: a_putv(state, flag_val) - sets state.flag = flag_val
Dictionary action_putv(Dictionary state, int flag_val) {
	Dictionary new_state = state.duplicate();
	new_state["flag"] = flag_val;
	return new_state;
}

// Action: a_getv(state, flag_val) - checks if state.flag == flag_val
Variant action_getv(Dictionary state, int flag_val) {
	if (state.has("flag") && int(state["flag"]) == flag_val) {
		// Return state unchanged (or empty dict to reset)
		return state.duplicate();
	}
	// Precondition not met, return false to indicate failure
	return false;
}

// Task methods for sample_test_1
// task_method_1 has 3 methods: task_method_1_1, task_method_1_2, task_method_1_3
Variant task_method_1_1(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_transfer_flag");
	task1.push_back(0);
	task1.push_back(1);
	Array task2;
	task2.push_back("action_transfer_flag");
	task2.push_back(1);
	task2.push_back(2);
	Array task3;
	task3.push_back("action_transfer_flag");
	task3.push_back(3);
	task3.push_back(4);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	subtasks.push_back(task3);
	return subtasks;
}

Variant task_method_1_2(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_transfer_flag");
	task1.push_back(0);
	task1.push_back(1);
	Array task2;
	task2.push_back("action_transfer_flag");
	task2.push_back(1);
	task2.push_back(2);
	Array task3;
	task3.push_back("action_transfer_flag");
	task3.push_back(2);
	task3.push_back(3);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	subtasks.push_back(task3);
	return subtasks;
}

Variant task_method_1_3(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_transfer_flag");
	task1.push_back(0);
	task1.push_back(1);
	Array task2;
	task2.push_back("action_transfer_flag");
	task2.push_back(1);
	task2.push_back(2);
	Array task3;
	task3.push_back("action_transfer_flag");
	task3.push_back(2);
	task3.push_back(3);
	Array task4;
	task4.push_back("action_transfer_flag");
	task4.push_back(3);
	task4.push_back(4);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	subtasks.push_back(task3);
	subtasks.push_back(task4);
	return subtasks;
}

// task_method_2 has 2 methods: task_method_2_1, task_method_2_2
Variant task_method_2_1(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_transfer_flag");
	task1.push_back(3);
	task1.push_back(4);
	Array task2;
	task2.push_back("action_transfer_flag");
	task2.push_back(4);
	task2.push_back(5);
	Array task3;
	task3.push_back("action_transfer_flag");
	task3.push_back(5);
	task3.push_back(6);
	Array task4;
	task4.push_back("action_transfer_flag");
	task4.push_back(6);
	task4.push_back(7);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	subtasks.push_back(task3);
	subtasks.push_back(task4);
	return subtasks;
}

Variant task_method_2_2(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_transfer_flag");
	task1.push_back(4);
	task1.push_back(5);
	Array task2;
	task2.push_back("action_transfer_flag");
	task2.push_back(5);
	task2.push_back(6);
	Array task3;
	task3.push_back("action_transfer_flag");
	task3.push_back(6);
	task3.push_back(7);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	subtasks.push_back(task3);
	return subtasks;
}

// Task methods for backtracking_test
Variant task_method_m_err(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_putv");
	task1.push_back(0);
	Array task2;
	task2.push_back("action_getv");
	task2.push_back(1);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	return subtasks;
}

Variant task_method_m0(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_putv");
	task1.push_back(0);
	Array task2;
	task2.push_back("action_getv");
	task2.push_back(0);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	return subtasks;
}

Variant task_method_m1(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_putv");
	task1.push_back(1);
	Array task2;
	task2.push_back("action_getv");
	task2.push_back(1);
	subtasks.push_back(task1);
	subtasks.push_back(task2);
	return subtasks;
}

Variant task_method_m_need0(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_getv");
	task1.push_back(0);
	subtasks.push_back(task1);
	return subtasks;
}

Variant task_method_m_need1(Dictionary state) {
	Array subtasks;
	Array task1;
	task1.push_back("action_getv");
	task1.push_back(1);
	subtasks.push_back(task1);
	return subtasks;
}

} // namespace IPyHOPTestDomain

// Implementations of IPyHOPTestDomainCallable static methods
inline Variant IPyHOPTestDomainCallable::action_transfer_flag(Dictionary p_state, int p_flag_key_1, int p_flag_key_2) {
	return IPyHOPTestDomain::action_transfer_flag(p_state, p_flag_key_1, p_flag_key_2);
}

inline Dictionary IPyHOPTestDomainCallable::action_putv(Dictionary p_state, int p_flag_val) {
	return IPyHOPTestDomain::action_putv(p_state, p_flag_val);
}

inline Variant IPyHOPTestDomainCallable::action_getv(Dictionary p_state, int p_flag_val) {
	return IPyHOPTestDomain::action_getv(p_state, p_flag_val);
}

inline Variant IPyHOPTestDomainCallable::task_method_1_1(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_1_1(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_1_2(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_1_2(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_1_3(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_1_3(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_2_1(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_2_1(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_2_2(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_2_2(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_m_err(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_m_err(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_m0(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_m0(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_m1(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_m1(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_m_need0(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_m_need0(p_state);
}

inline Variant IPyHOPTestDomainCallable::task_method_m_need1(Dictionary p_state) {
	return IPyHOPTestDomain::task_method_m_need1(p_state);
}
