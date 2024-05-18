#include "multigoal.h"

void Multigoal::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_to_string"), &Multigoal::_to_string);
    ClassDB::bind_method(D_METHOD("get_state"), &Multigoal::get_state);
    ClassDB::bind_method(D_METHOD("set_state", "value"), &Multigoal::set_state);
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "state"), "set_state", "get_state");

    ClassDB::bind_method(D_METHOD("_init", "multigoal_name", "state_variables"), &Multigoal::_init);
    ClassDB::bind_method(D_METHOD("state_vars"), &Multigoal::state_vars);
}

String Multigoal::_to_string() {
    return get_name();
}

Dictionary Multigoal::get_state() const {
    return _state;
}

void Multigoal::set_state(Dictionary value) {
    _state = value;
}

void Multigoal::_init(String multigoal_name, Dictionary state_variables) {
    set_name(multigoal_name);
    _state = state_variables;
}

Array Multigoal::state_vars() {
    return _state.keys();
}
