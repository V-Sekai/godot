// Port of ParallelCommits from Elixir to Godot

#pragma once

#include "core/io/resource.h"
#include "core/object/object.h"
#include "planner_consensus_result.h"
#include "planner_global_state.h"
#include "scene/main/node.h"
#include "core/templates/hash_map.h"

class PlannerParallelCommits : public Node {
    GDCLASS(PlannerParallelCommits, Node);

public:
    enum RecordStatus {
        PENDING,
        STAGING,
        COMMITTED,
        ABORTED
    };

    struct TransactionRecord {
        RecordStatus status;
        int64_t epoch;
        int64_t ts;
    };

private:
    int64_t logical_time;
    int64_t counter;
    TransactionRecord record;
    HashMap<String, Dictionary> intent_writes;
    HashMap<String, int64_t> tscache;
    bool commit_ack = false;
    Callable callback;
    ObjectID callback_instance_id;

public:
    PlannerParallelCommits();
    Error init_system();
    Ref<PlannerConsensusResult> submit_operation(Dictionary p_operation);
    Ref<PlannerGlobalState> get_global_state();
    void set_callback(Callable p_callback) { callback = p_callback; }
    Callable get_callback() const { return callback; }
    String generate_uuidv7();
    void broadcast_operation(Dictionary p_operation);
    void receive_operation(Dictionary p_operation);
    void _process(double delta);  // For interpolation
    void _ready();  // Setup networking

    void set_callback_object(Object *p_object) { callback_instance_id = p_object ? p_object->get_instance_id() : ObjectID(); }
    Object *get_callback_object() const { return callback_instance_id.is_valid() ? ObjectDB::get_instance(callback_instance_id) : nullptr; }

    void rpc_broadcast_operation(Dictionary p_operation);
    void rpc_receive_operation(Dictionary p_operation);

    // Sync properties
    int64_t get_logical_time() const { return logical_time; }
    void set_logical_time(int64_t p_time) { logical_time = p_time; }
    Dictionary get_record() const;
    void set_record(Dictionary p_record);
    Dictionary get_intent_writes() const;
    void set_intent_writes(Dictionary p_writes);
    Dictionary get_tscache() const;
    void set_tscache(Dictionary p_cache);

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("init_system"), &PlannerParallelCommits::init_system);
        ClassDB::bind_method(D_METHOD("submit_operation", "operation"), &PlannerParallelCommits::submit_operation);
        ClassDB::bind_method(D_METHOD("get_global_state"), &PlannerParallelCommits::get_global_state);
        ClassDB::bind_method(D_METHOD("get_record"), &PlannerParallelCommits::get_record);
        ClassDB::bind_method(D_METHOD("set_record", "record"), &PlannerParallelCommits::set_record);
        ClassDB::bind_method(D_METHOD("get_intent_writes"), &PlannerParallelCommits::get_intent_writes);
        ClassDB::bind_method(D_METHOD("set_intent_writes", "writes"), &PlannerParallelCommits::set_intent_writes);
        ClassDB::bind_method(D_METHOD("get_tscache"), &PlannerParallelCommits::get_tscache);
        ClassDB::bind_method(D_METHOD("set_tscache", "cache"), &PlannerParallelCommits::set_tscache);
        ClassDB::bind_method(D_METHOD("_process", "delta"), &PlannerParallelCommits::_process);
        ClassDB::bind_method(D_METHOD("_ready"), &PlannerParallelCommits::_ready);
        ClassDB::bind_method(D_METHOD("broadcast_operation", "operation"), &PlannerParallelCommits::broadcast_operation);
        ClassDB::bind_method(D_METHOD("receive_operation", "operation"), &PlannerParallelCommits::receive_operation);
        ClassDB::bind_method(D_METHOD("set_callback_object", "object"), &PlannerParallelCommits::set_callback_object);
        ClassDB::bind_method(D_METHOD("get_callback_object"), &PlannerParallelCommits::get_callback_object);
    }
};