// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// multigoal.h
// SPDX-License-Identifier: MIT

#pragma once

#include "core/io/resource.h"
#include "core/variant/dictionary.h"

class Multigoal : public Resource {
    GDCLASS(Multigoal, Resource);

private:
    Dictionary _state;

public:
    String _to_string();
    Dictionary get_state() const;
    void set_state(Dictionary value);
    void _init(String multigoal_name, Dictionary state_variables);
    void display(String heading = "");
    Array state_vars();

protected:
    static void _bind_methods();
};
