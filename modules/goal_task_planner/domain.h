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
    Variant _m_verify_g(Dictionary state, String method, String state_var, String arg, Variant desired_val, int depth);
    static Dictionary _goals_not_achieved(Dictionary state, Ref<Multigoal> multigoal);
    Variant _m_verify_mg(Dictionary state, String method, Ref<Multigoal> multigoal, int depth);
    void display();

protected:
    static void _bind_methods();
};
