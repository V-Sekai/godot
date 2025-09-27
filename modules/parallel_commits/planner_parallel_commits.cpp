// Implementation of PlannerParallelCommits

#include "planner_parallel_commits.h"
#include "core/os/os.h"
#include "core/variant/variant.h"
#include "core/crypto/crypto_core.h"

PlannerParallelCommits::PlannerParallelCommits() : record{ PENDING, 0, 0 } {}

Error PlannerParallelCommits::init_system() {
    // Initialize record to pending
    record = { PENDING, 0, 0 };
    intent_writes.clear();
    tscache.clear();
    commit_ack = false;
    if (callback_instance_id.is_valid()) {
        Object *obj = ObjectDB::get_instance(callback_instance_id);
        if (obj) {
            obj->call("_on_system_initialized");
        }
    }
    return OK;
}

Ref<PlannerConsensusResult> PlannerParallelCommits::submit_operation(Dictionary p_operation) {
    // Simulate committer process from TLA+
    // BeginTxnEpoch
    record.epoch += 1;
    record.ts += 1;

    // Assume parallel_keys = all keys, pipelined_keys = empty for simplicity
    Array keys = p_operation.get("keys", Array()); // Assume operation has keys

    // PipelineWrites: for simplicity, assume no pipelined writes

    // ParallelCommit: set record to staging
    record.status = STAGING;

    // Write intents
    for (int i = 0; i < keys.size(); i++) {
        String key = keys[i];
        Dictionary intent;
        intent["epoch"] = record.epoch;
        intent["ts"] = record.ts;
        intent["resolved"] = false;
        intent_writes[key] = intent;
        tscache[key] = record.ts;
    }

    // Check if implicitly committed (all intents written)
    bool implicitly_committed = true;
    for (int i = 0; i < keys.size(); i++) {
        String key = keys[i];
        if (!intent_writes.has(key)) {
            implicitly_committed = false;
            break;
        }
        Variant intent = intent_writes[key];
        if (intent.get_type() == Variant::DICTIONARY) {
            Dictionary intent_dict = intent;
            if (intent_dict["resolved"]) {
                implicitly_committed = false;
                break;
            }
        }
    }

    if (implicitly_committed) {
        // Move to committed
        record.status = COMMITTED;
        commit_ack = true;
    }

    // Generate ID and result
    String transaction_id;
    Error err = CryptoCore::generate_uuidv7(transaction_id);
    if (err != OK) {
        print_line("Failed to generate UUIDv7: " + itos(err));
        return Ref<PlannerConsensusResult>(); // or handle error
    }

    Ref<PlannerConsensusResult> result;
    result.instantiate();
    result->set_operation_id(transaction_id);
    result->set_agreed_at(OS::get_singleton()->get_unix_time() * 1000);
    Array participants;
    participants.push_back("node_1");
    result->set_participants(participants);

    print_line("ParallelCommits operation submitted [" + transaction_id + "]: " + String(Variant(p_operation)));
    if (callback.is_valid()) {
        callback.call(result);
    }

    return result;
}

Ref<PlannerGlobalState> PlannerParallelCommits::get_global_state() {
    // Port of get_global_state from Elixir
    Ref<PlannerGlobalState> state;
    state.instantiate();
    Dictionary record_dict;
    record_dict["status"] = record.status == PENDING ? "pending" : record.status == STAGING ? "staging" : record.status == COMMITTED ? "committed" : "aborted";
    record_dict["epoch"] = record.epoch;
    record_dict["ts"] = record.ts;
    state->set_record(record_dict);
    state->set_intent_writes(get_intent_writes());
    state->set_tscache(get_tscache());
    state->set_commit_ack(commit_ack);
    Dictionary hlc_dict;
    hlc_dict["l"] = 0; // Placeholder
    hlc_dict["c"] = 0;
    state->set_hlc(hlc_dict);

    return state;
}

Dictionary PlannerParallelCommits::get_intent_writes() const {
    Dictionary d;
    for (const auto &pair : intent_writes) {
        d[pair.key] = pair.value;
    }
    return d;
}

void PlannerParallelCommits::set_intent_writes(Dictionary p_writes) {
    intent_writes.clear();
    for (const Variant &key_var : p_writes.keys()) {
        String key = key_var;
        Dictionary intent_dict = p_writes[key_var];
        intent_writes[key] = intent_dict;
    }
}

Dictionary PlannerParallelCommits::get_tscache() const {
    Dictionary d;
    for (const auto &pair : tscache) {
        d[pair.key] = pair.value;
    }
    return d;
}

void PlannerParallelCommits::set_tscache(Dictionary p_cache) {
    tscache.clear();
    for (const Variant &key_var : p_cache.keys()) {
        String key = key_var;
        tscache[key] = p_cache[key_var];
    }
}