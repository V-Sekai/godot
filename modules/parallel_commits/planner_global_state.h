#pragma once

#include "core/io/resource.h"

class PlannerGlobalState : public Resource {
    GDCLASS(PlannerGlobalState, Resource);

private:
    Dictionary record;
    Dictionary intent_writes;
    Dictionary tscache;
    bool commit_ack;
    Dictionary hlc;

public:
    void set_record(Dictionary p_record) { record = p_record; }
    Dictionary get_record() const { return record; }
    void set_intent_writes(Dictionary p_writes) { intent_writes = p_writes; }
    Dictionary get_intent_writes() const { return intent_writes; }
    void set_tscache(Dictionary p_cache) { tscache = p_cache; }
    Dictionary get_tscache() const { return tscache; }
    void set_commit_ack(bool p_ack) { commit_ack = p_ack; }
    bool get_commit_ack() const { return commit_ack; }
    void set_hlc(Dictionary p_hlc) { hlc = p_hlc; }
    Dictionary get_hlc() const { return hlc; }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_record", "record"), &PlannerGlobalState::set_record);
        ClassDB::bind_method(D_METHOD("get_record"), &PlannerGlobalState::get_record);
        ClassDB::bind_method(D_METHOD("set_intent_writes", "writes"), &PlannerGlobalState::set_intent_writes);
        ClassDB::bind_method(D_METHOD("get_intent_writes"), &PlannerGlobalState::get_intent_writes);
        ClassDB::bind_method(D_METHOD("set_tscache", "cache"), &PlannerGlobalState::set_tscache);
        ClassDB::bind_method(D_METHOD("get_tscache"), &PlannerGlobalState::get_tscache);
        ClassDB::bind_method(D_METHOD("set_commit_ack", "ack"), &PlannerGlobalState::set_commit_ack);
        ClassDB::bind_method(D_METHOD("get_commit_ack"), &PlannerGlobalState::get_commit_ack);
        ClassDB::bind_method(D_METHOD("set_hlc", "hlc"), &PlannerGlobalState::set_hlc);
        ClassDB::bind_method(D_METHOD("get_hlc"), &PlannerGlobalState::get_hlc);
    }
};