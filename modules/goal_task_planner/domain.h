// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// domain.h
// SPDX-License-Identifier: MIT

#pragma once

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

    void set_verbose(int value) { verbose = value; }
    void set_action_dict(Dictionary value) { _action_dict = value; }
    void set_task_method_dict(Dictionary value) { _task_method_dict = value; }
    void set_unigoal_method_dict(Dictionary value) { _unigoal_method_dict = value; }
    void set_multigoal_method_list(Array value) { _multigoal_method_list = value; }

    int get_verbose() const { return verbose; }
    Dictionary get_action_dict() const { return _action_dict; }
    Dictionary get_task_method_dict() const { return _task_method_dict; }
    Dictionary get_unigoal_method_dict() const { return _unigoal_method_dict; }
    Array get_multigoal_method_list() const { return _multigoal_method_list; }

public:
    Variant _m_verify_g(Dictionary state, String method, String state_var, String arg, Variant desired_val, int depth);
    static Dictionary _goals_not_achieved(Dictionary state, Ref<Multigoal> multigoal);
    Variant _m_verify_mg(Dictionary state, String method, Ref<Multigoal> multigoal, int depth);
    void display();

protected:
    static void _bind_methods();
};
