// temporal_constraint.cpp

#include "temporal_constraint.h"

void TemporalConstraint::_init(int start, int end, int duration, int qualifier, String resource) {
    time_interval = Vector2i(start, end);
    this->duration = duration;
    temporal_qualifier =  TemporalQualifier(qualifier);
    resource_name = resource;
}

bool TemporalConstraint::overlaps_with(Ref<TemporalConstraint> other) {
    if (resource_name != other->resource_name)
        return false;

    int self_start, self_end, other_start, other_end;

    // Calculate the effective start and end times based on the temporal qualifier
    switch (temporal_qualifier) {
        case AT_START:
            self_start = time_interval.x;
            self_end = time_interval.x + duration;
            break;
        case AT_END:
            self_start = time_interval.y - duration;
            self_end = time_interval.y;
            break;
        case OVERALL:
            self_start = time_interval.x;
            self_end = time_interval.y;
            break;
    }

    switch (other->temporal_qualifier) {
        case AT_START:
            other_start = other->time_interval.x;
            other_end = other->time_interval.x + other->duration;
            break;
        case AT_END:
            other_start = other->time_interval.y - other->duration;
            other_end = other->time_interval.y;
            break;
        case OVERALL:
            other_start = other->time_interval.x;
            other_end = other->time_interval.y;
            break;
    }

    // Check if intervals overlap
    return self_start < other_end && other_start < self_end;
}

String TemporalConstraint::_to_string() {
    Dictionary dict;
    dict["resource_name"] = resource_name;
    dict["time_interval"] = time_interval;
    dict["duration"] = duration;
    dict["temporal_qualifier"] = temporal_qualifier;

    return String(dict);
}

void TemporalConstraint::_bind_methods() {
    // ClassDB::bind_method(D_METHOD("_init", "start", "end", "duration", "qualifier", "resource"), &TemporalConstraint::_init);
    ClassDB::bind_method(D_METHOD("overlaps_with", "other"), &TemporalConstraint::overlaps_with);
    ClassDB::bind_method(D_METHOD("_to_string"), &TemporalConstraint::_to_string);

    ClassDB::bind_method(D_METHOD("get_time_interval"), &TemporalConstraint::get_time_interval);
    ClassDB::bind_method(D_METHOD("set_time_interval", "interval"), &TemporalConstraint::set_time_interval);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "time_interval"), "set_time_interval", "get_time_interval");

    ClassDB::bind_method(D_METHOD("get_duration"), &TemporalConstraint::get_duration);
    ClassDB::bind_method(D_METHOD("set_duration", "duration"), &TemporalConstraint::set_duration);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "duration"), "set_duration", "get_duration");

    ClassDB::bind_method(D_METHOD("get_temporal_qualifier"), &TemporalConstraint::get_temporal_qualifier);
    ClassDB::bind_method(D_METHOD("set_temporal_qualifier", "qualifier"), &TemporalConstraint::set_temporal_qualifier);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "temporal_qualifier"), "set_temporal_qualifier", "get_temporal_qualifier");
}
