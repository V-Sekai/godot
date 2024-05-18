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
	Dictionary _action_dict;
	Dictionary _task_method_dict;
	Dictionary _unigoal_method_dict;
	Array _multigoal_method_list;
	// Ref<SimpleTemporalNetwork> stn;

public:
	void set_verbose(int p_value) { verbose = p_value; }
	void set_action_dict(Dictionary p_value) { _action_dict = p_value; }
	void set_task_method_dict(Dictionary p_value) { _task_method_dict = p_value; }
	void set_unigoal_method_dict(Dictionary p_value) { _unigoal_method_dict = p_value; }
	void set_multigoal_method_list(Array p_value) { _multigoal_method_list = p_value; }

	int get_verbose() const { return verbose; }
	Dictionary get_action_dict() const { return _action_dict; }
	Dictionary get_task_method_dict() const { return _task_method_dict; }
	Dictionary get_unigoal_method_dict() const { return _unigoal_method_dict; }
	Array get_multigoal_method_list() const { return _multigoal_method_list; }

public:
	Variant _m_verify_g(Dictionary p_state, String p_method, String p_state_var, String p_arguments, Variant p_desired_values, int p_depth);
	static Dictionary _goals_not_achieved(Dictionary p_state, Ref<Multigoal> p_multigoal);
	Variant _m_verify_mg(Dictionary p_state, String p_method, Ref<Multigoal> p_multigoal, int p_depth);

protected:
	static void _bind_methods();
};

#endif // DOMAIN_H
