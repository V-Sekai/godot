
#pragma once

#include "core/io/resource.h"

class PlannerConsensusResult : public Resource {
    GDCLASS(PlannerConsensusResult, Resource);

private:
    String operation_id;
    int64_t agreed_at;
    Array participants;

public:
    void set_operation_id(String p_id) { operation_id = p_id; }
    String get_operation_id() const { return operation_id; }
    void set_agreed_at(int64_t p_time) { agreed_at = p_time; }
    int64_t get_agreed_at() const { return agreed_at; }
    void set_participants(Array p_participants) { participants = p_participants; }
    Array get_participants() const { return participants; }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_operation_id", "id"), &PlannerConsensusResult::set_operation_id);
        ClassDB::bind_method(D_METHOD("get_operation_id"), &PlannerConsensusResult::get_operation_id);
        ClassDB::bind_method(D_METHOD("set_agreed_at", "time"), &PlannerConsensusResult::set_agreed_at);
        ClassDB::bind_method(D_METHOD("get_agreed_at"), &PlannerConsensusResult::get_agreed_at);
        ClassDB::bind_method(D_METHOD("set_participants", "participants"), &PlannerConsensusResult::set_participants);
        ClassDB::bind_method(D_METHOD("get_participants"), &PlannerConsensusResult::get_participants);
    }
};