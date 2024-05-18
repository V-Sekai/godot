// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// temporal_constraint.h
// SPDX-License-Identifier: MIT

#pragma once

#include "core/io/resource.h"
#include "core/math/vector2i.h"

class TemporalConstraint : public Resource {
    GDCLASS(TemporalConstraint, Resource);

public:
    enum TemporalQualifier {
        AT_START,
        AT_END,
        OVERALL,
    };

private:
    Vector2i time_interval;
    int duration;
    TemporalQualifier temporal_qualifier;
    String resource_name;

public:
    void _init(int start, int end, int duration, int qualifier, String resource);
    bool overlaps_with(Ref<TemporalConstraint> other);
    String _to_string();

    Vector2i get_time_interval() const { return time_interval; }
    int get_duration() const { return duration; }
    int get_temporal_qualifier() const { return temporal_qualifier; }

    void set_time_interval(Vector2i interval) { time_interval = interval; }
    void set_duration(int dur) { duration = dur; }
    void set_temporal_qualifier(int qualifier) { temporal_qualifier = TimeQualifier(qualifier); }

protected:
    static void _bind_methods();
};
