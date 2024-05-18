/**************************************************************************/
/*  domain.h                                                              */
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

#ifndef DOMAIN_H
#define DOMAIN_H
// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// domain.h
// SPDX-License-Identifier: MIT

#include "core/io/resource.h"
#include "multigoal.h"

class Domain : public Resource {
	GDCLASS(Domain, Resource);

private:
	int verbose = 0;
	Dictionary action_dictionary;
	Dictionary task_method_dictionary;
	Dictionary unigoal_method_dictionary;
	Array multigoal_method_list;
	// Ref<SimpleTemporalNetwork> stn;

public:
	void set_verbose(int p_value) { verbose = p_value; }
	void set_action_dictionary(Dictionary p_value) { action_dictionary = p_value; }
	void set_task_method_dictionary(Dictionary p_value) { task_method_dictionary = p_value; }
	void set_unigoal_method_dictionary(Dictionary p_value) { unigoal_method_dictionary = p_value; }
	void set_multigoal_method_list(Array p_value) { multigoal_method_list = p_value; }

	int get_verbose() const { return verbose; }
	Dictionary get_action_dictionary() const { return action_dictionary; }
	Dictionary get_task_method_dictionary() const { return task_method_dictionary; }
	Dictionary get_unigoal_method_dictionary() const { return unigoal_method_dictionary; }
	Array get_multigoal_method_list() const { return multigoal_method_list; }
	// void Plan::print_simple_temporal_network(Ref<Domain> domain) {
	// 	if (domain == nullptr) {
	// 		domain = current_domain;
	// 	}

	// 	if (domain->stn) {
	// 		std::cout << "-- Simple Temporal Network: " << domain->stn.to_dictionary() << std::endl;
	// 	} else {
	// 		std::cout << "-- There is no Simple Temporal Network --" << std::endl;
	// 	}
	// }
	void print_domain() const {
		print_line(vformat("Domain name: %s", get_name()));
		print_actions();
		print_methods();
		// print_simple_temporal_network(p_domain);
	}

	void print_actions() const {
		if (get_action_dictionary().is_empty()) {
			print_line("-- There are no actions --");
			return;
		}
		String actions = "-- Actions: ";

		// Create an iterator for the action_dict
		Array keys = get_action_dictionary().keys();
		for (int i = 0; i < keys.size(); ++i) {
			if (i != 0) {
				actions += ", ";
			}
			actions += String(keys[i]);
		}
		print_line(actions);
	}

	void print_task_methods() const {
		if (get_task_method_dictionary().is_empty()) {
			print_line("-- There are no task methods --");
			return;
		}
		print_line("\nTask name:         Relevant task methods:");
		print_line("---------------    ----------------------");

		String string_array;
		Array keys = get_task_method_dictionary().keys();
		for (int i = 0; i < keys.size(); ++i) {
			string_array += String(keys[i]) + ", ";
		}
		print_line(string_array.substr(0, string_array.length() - 2)); // Remove last comma and space

		print_line("");
	}

	void print_unigoal_methods() const {
		if (get_unigoal_method_dictionary().is_empty()) {
			print_line("-- There are no unigoal methods --");
			return;
		}
		print_line("Blackboard var name:    Relevant unigoal methods:");
		print_line("---------------    -------------------------");
		Array keys = get_unigoal_method_dictionary().keys();
		for (int j = 0; j < keys.size(); ++j) {
			String string_array;
			Array methods = get_unigoal_method_dictionary()[keys[j]];
			for (int i = 0; i < methods.size(); ++i) {
				String m = methods[i];
				string_array += m + ", ";
			}
			print_line(string_array.substr(0, string_array.length() - 2)); // Remove last comma and space
		}

		print_line("");
	}

	void print_multigoal_methods() const {
		if (get_multigoal_method_list().is_empty()) {
			print_line("-- There are no multigoal methods --");
			return;
		}
		String string_array;
		Array methods = get_multigoal_method_list();
		for (int i = 0; i < methods.size(); ++i) {
			Variant m = methods[i];
			string_array += String(m) + ", ";
		}
		print_line("-- Multigoal methods: " + string_array.substr(0, string_array.length() - 2)); // Remove last comma and space
	}

	void print_methods() const {
		print_task_methods();
		print_unigoal_methods();
		print_multigoal_methods();
	}

public:
	Variant _m_verify_g(Dictionary p_state, String p_method, String p_state_var, String p_arguments, Variant p_desired_values, int p_depth);
	static Dictionary _goals_not_achieved(Dictionary p_state, Ref<Multigoal> p_multigoal);
	Variant _m_verify_mg(Dictionary p_state, String p_method, Ref<Multigoal> p_multigoal, int p_depth);

protected:
	static void _bind_methods();
};

#endif // DOMAIN_H
