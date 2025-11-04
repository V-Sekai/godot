/**************************************************************************/
/*  plan.cpp                                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "plan.h"

#include "core/variant/callable.h"
#include "core/variant/typed_array.h"
#include "core/os/os.h"
#include "core/crypto/crypto_core.h"
#include "core/io/json.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/string/ustring.h"

#include "modules/sqlite/src/godot_sqlite.h"
#include "domain.h"
#include "multigoal.h"
#include "graph_operations.h"
#include "backtracking.h"
#include "stn_constraints.h"

int PlannerPlan::get_verbose() const {
	return verbose;
}

TypedArray<PlannerDomain> PlannerPlan::get_domains() const {
	return domains;
}

Ref<PlannerDomain> PlannerPlan::get_current_domain() const {
	return current_domain;
}

void PlannerPlan::set_verbose(int p_verbose) {
	verbose = p_verbose;
}

void PlannerPlan::set_domains(TypedArray<PlannerDomain> p_domain) {
	domains = p_domain;
}

Variant PlannerPlan::_apply_action_and_continue(Dictionary p_state, Array p_first_task, Array p_todo_list, Array p_plan, int p_depth) {
	Callable action = current_domain->action_dictionary[p_first_task[0]];

	if (verbose >= 2) {
		Array action_info = p_first_task.slice(1);
		action_info.insert(0, action.get_method());
		print_line("Depth: " + itos(p_depth) + ", Action: " + _item_to_string(action_info));
	}

	Array arguments = p_first_task.slice(1);
	arguments.insert(0, p_state);
	Variant new_state = action.callv(arguments);

	if (new_state) {
		if (verbose >= 3) {
			print_line("Intermediate computation: Action applied successfully.");
			print_line("New state: " + String(new_state));
		}
		Array new_plan = p_plan;
		new_plan.push_back(p_first_task);
		return _seek_plan(new_state, p_todo_list, new_plan, p_depth + 1);
	}

	if (verbose >= 3) {
		print_line("Intermediate computation: Failed to apply action. The new state is not valid.");
		print_line("New state: " + String(new_state));
		print_line("Task: ");
		for (int i = 0; i < p_first_task.size(); ++i) {
			print_line(String(p_first_task[i]));
			print_line("State: " + _item_to_string(p_state));
		}

		if (verbose >= 2) {
			Array action_info = p_first_task.slice(1);
			action_info.insert(0, action.get_method());
			ERR_PRINT("Recursive call: Not applicable action: " + _item_to_string(action_info));
		}
	}
	return false;
}

Variant PlannerPlan::_refine_task_and_continue(const Dictionary p_state, const Array p_task1, const Array p_todo_list, const Array p_plan, const int p_depth) {
	Array relevant = current_domain->task_method_dictionary[p_task1[0]];
	if (verbose >= 3) {
		print_line("Depth: " + itos(p_depth) + ", Task " + _item_to_string(p_task1) + ", Todo List " + _item_to_string(p_todo_list) + ", Plan " + _item_to_string(p_plan));
	}
	for (int i = 0; i < relevant.size(); i++) {
		Callable method = relevant[i];
		Array arguments;
		arguments.push_back(p_state);
		Array argument_slices = p_task1.slice(1);
		arguments.append_array(argument_slices);
		if (verbose >= 2) {
			print_line("Depth: " + itos(p_depth) + ", Trying method: " + _item_to_string(method));
		}
		Variant result = method.callv(arguments);
		if (result.is_array()) {
			Array subgoals = result;
			Array todo_list;
			if (!subgoals.is_empty()) {
				todo_list.append_array(subgoals);
			}
			if (!p_todo_list.is_empty()) {
				todo_list.append_array(p_todo_list);
			}
			Variant plan = _seek_plan(p_state, todo_list, p_plan, p_depth + 1);
			if (plan.is_array()) {
				return plan;
			}
		} else {
			if (verbose >= 3) {
				ERR_PRINT("Not applicable");
			}
		}
	}

	if (verbose >= 2) {
		ERR_PRINT("Recursive call: Failed to achieve task: " + _item_to_string(p_task1));
	}

	return false;
}

Variant PlannerPlan::_refine_multigoal_and_continue(const Dictionary p_state, const Ref<PlannerMultigoal> p_first_goal, const Array p_todo_list, const Array p_plan, const int p_depth) {
	if (verbose >= 3) {
		print_line("Depth: " + itos(p_depth) + ", Multigoal: " + p_first_goal->get_name() + ": " + _item_to_string(p_first_goal));
	}

	Array relevant = current_domain->multigoal_method_list;

	if (verbose >= 3) {
		Array string_array;
		for (int i = 0; i < relevant.size(); i++) {
			print_line(String("Methods ") + String(relevant[i].call("get_method")));
		}
	}
	Array todo_list = p_todo_list;
	for (int i = 0; i < relevant.size(); i++) {
		if (verbose >= 2) {
			print_line("Depth: " + itos(p_depth) + ", Trying method: " + String(relevant[i].call("get_method")) + ": ");
		}
		Callable callable = relevant[i];
		Variant result = callable.call(p_state, p_first_goal);
		if (result.is_array()) {
			Array subgoals = result;
			Array subtodo_list;
			if (verbose >= 3) {
				print_line("Intermediate computation: Method applicable.");
				print_line("Depth: " + itos(p_depth) + ", Subgoals: " + _item_to_string(subgoals));
			}
			if (!subgoals.is_empty()) {
				subtodo_list.append_array(subgoals);
			}
			if (verify_goals) {
				subtodo_list.push_back(varray("_verify_mg", callable.get_method(), p_first_goal, p_depth, verbose));
			}
			if (!p_todo_list.is_empty()) {
				subtodo_list.append_array(p_todo_list);
			}
			todo_list.clear();
			todo_list = subtodo_list;
			Variant plan = _seek_plan(p_state, todo_list, p_plan, p_depth + 1);
			if (plan.is_array()) {
				return plan;
			}
		}
	}

	if (verbose >= 2) {
		ERR_PRINT("Recursive call: Failed to achieve multigoal: " + _item_to_string(p_first_goal));
	}

	return false;
}

Variant PlannerPlan::_refine_unigoal_and_continue(const Dictionary p_state, const Array p_goal1, const Array p_todo_list, const Array p_plan, const int p_depth) {
	if (verbose >= 3) {
		String goals_list = vformat("Depth: %d, Goals: %s", p_depth, _item_to_string(p_goal1));
	}

	String state_variable_name = p_goal1[0];
	String argument = p_goal1[1];
	Variant value = p_goal1[2];

	Dictionary state_variable = p_state[state_variable_name];

	if (state_variable[argument] == value) {
		if (verbose >= 3) {
			print_line("Intermediate computation: Goal already achieved.");
		}
		return _seek_plan(p_state, p_todo_list, p_plan, p_depth + 1);
	}

	Array relevant = current_domain->unigoal_method_dictionary[state_variable_name];
	if (verbose >= 3) {
		print_line("Methods: " + _item_to_string(relevant));
	}
	Array todo_list = p_todo_list;
	for (int i = 0; i < relevant.size(); i++) {
		Callable method = relevant[i];
		if (verbose >= 2) {
			print_line("Depth: " + itos(p_depth) + ", Trying method: " + _item_to_string(method));
		}
		Variant result = method.call(p_state, argument, value);
		if (result.is_array()) {
			Array subgoals = result;
			Array subtodo_list;
			if (!subgoals.is_empty()) {
				subtodo_list.append_array(subgoals);
			}
			if (verify_goals) {
				subtodo_list.push_back(varray("_verify_g", method.get_method(), state_variable_name, argument, value, p_depth, verbose));
			}
			if (!todo_list.is_empty()) {
				subtodo_list.append_array(todo_list);
			}
			todo_list.clear();
			todo_list = subtodo_list;
			if (verbose >= 3) {
				print_line("Depth: " + itos(p_depth) + ", Seeking todo list " + _item_to_string(todo_list));
			}
			Variant plan = _seek_plan(p_state, todo_list, p_plan, p_depth + 1);
			if (plan.is_array()) {
				return plan;
			}
		} else {
			if (verbose >= 3) {
				ERR_PRINT("Not applicable");
			}
		}
	}

	if (verbose >= 2) {
		ERR_PRINT(vformat("Recursive call: Failed to achieve goal: %s", _item_to_string(p_goal1)));
	}

	return false;
}

Variant PlannerPlan::find_plan(Dictionary p_state, Array p_todo_list) {
	if (verbose >= 1) {
		print_line("verbose=" + itos(verbose) + ":");
		print_line("    state = " + _item_to_string(p_state) + "\n    todo_list = " + _item_to_string(p_todo_list));
	}

	Variant result = _seek_plan(p_state, p_todo_list, Array(), 0);

	if (verbose >= 1) {
		print_line("result = " + _item_to_string(result));
	}

	return result;
}

Variant PlannerPlan::_seek_plan(Dictionary p_state, Array p_todo_list, Array p_plan, int p_depth) {
	if (verbose >= 2) {
		print_line("Depth: " + itos(p_depth) + ", Todo List: " + _item_to_string(p_todo_list));
	}

	if (p_todo_list.is_empty()) {
		if (verbose >= 3) {
			print_line("Depth: " + itos(p_depth) + " no more tasks or goals, return plan: " + _item_to_string(p_plan));
		}
		return p_plan;
	}
	Variant todo_item = p_todo_list.front();
	p_todo_list = p_todo_list.slice(1);
	if (Object::cast_to<PlannerMultigoal>(todo_item)) {
		return _refine_multigoal_and_continue(p_state, todo_item, p_todo_list, p_plan, p_depth);
	} else if (todo_item.is_array()) {
		Array item = todo_item;
		Dictionary actions = current_domain->action_dictionary;
		Dictionary tasks = current_domain->task_method_dictionary;
		Dictionary unigoals = current_domain->unigoal_method_dictionary;
		Variant item_name = item.front();
		if (actions.has(item_name)) {
			return _apply_action_and_continue(p_state, item, p_todo_list, p_plan, p_depth);
		} else if (tasks.has(item_name)) {
			return _refine_task_and_continue(p_state, item, p_todo_list, p_plan, p_depth);
		} else if (unigoals.has(item_name)) {
			return _refine_unigoal_and_continue(p_state, item, p_todo_list, p_plan, p_depth);
		}
	}
	return false;
}

String PlannerPlan::_item_to_string(Variant p_item) {
	return String(p_item);
}

Dictionary PlannerPlan::run_lazy_lookahead(Dictionary p_state, Array p_todo_list, int p_max_tries) {
	if (verbose >= 1) {
		print_line(vformat("run_lazy_lookahead: verbose = %s, max_tries = %s", verbose, p_max_tries));
		print_line(vformat("Initial state: %s", p_state.keys()));
		print_line(vformat("To do: %s", p_todo_list));
	}

	Dictionary ordinals;
	ordinals[1] = "st";
	ordinals[2] = "nd";
	ordinals[3] = "rd";

	for (int tries = 1; tries <= p_max_tries; tries++) {
		if (verbose >= 1) {
			print_line(vformat("run_lazy_lookahead: %sth call to find_plan: %s", tries, ordinals.get(tries, "")));
		}

		Variant plan = find_plan(p_state, p_todo_list);
		if (plan == Variant(false)) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("run_lazy_lookahead: find_plan has failed after %s calls.", tries));
			}
			return p_state;
		}

		if (plan.is_array() && Array(plan).is_empty()) {
			if (verbose >= 1) {
				print_line(vformat("run_lazy_lookahead: Empty plan => success\nafter %s calls to find_plan.", tries));
			}
			if (verbose >= 2) {
				print_line(vformat("run_lazy_lookahead: final state %s", p_state));
			}
			return p_state;
		}

		if (plan.is_array()) {
			Array action_list = plan;
			for (int i = 0; i < action_list.size(); i++) {
				Array action = action_list[i];
				Callable action_name = current_domain->action_dictionary[action[0]];
				if (verbose >= 1) {
					String action_arguments;
					Array actions = action.slice(1, action.size());
					for (Variant element : actions) {
						action_arguments += String(" ") + String(element);
					}
					print_line(vformat("run_lazy_lookahead: Task: %s, %s", action_name.get_method(), action_arguments));
				}

				Dictionary new_state = _apply_task_and_continue(p_state, action_name, action.slice(1, action.size()));
				if (!new_state.is_empty()) {
					if (verbose >= 2) {
						print_line(new_state);
					}
					p_state = new_state;
				} else {
					if (verbose >= 1) {
						ERR_PRINT(vformat("run_lazy_lookahead: WARNING: action %s failed; will call find_plan.", action_name));
					}
					break;
				}
			}
		}

		if (verbose >= 1 && !p_state.is_empty()) {
			print_line("RunLazyLookahead> Plan ended; will call find_plan again.");
		}
	}

	if (verbose >= 1) {
		ERR_PRINT("run_lazy_lookahead: Too many tries, giving up.");
	}
	if (verbose >= 2) {
		print_line(vformat("run_lazy_lookahead: final state %s", p_state));
	}

	return p_state;
}

Variant PlannerPlan::_apply_task_and_continue(Dictionary p_state, Callable p_command, Array p_arguments) {
	if (verbose >= 3) {
		print_line(vformat("_apply_task_and_continue %s, args = %s", p_command.get_method(), _item_to_string(p_arguments)));
	}
	Array argument;
	argument.push_back(p_state);
	argument.append_array(p_arguments);
	Variant next_state = p_command.callv(argument);
	if (!next_state) {
		if (verbose >= 3) {
			print_line(vformat("Not applicable command %s", argument));
		}
		return false;
	}

	if (verbose >= 3) {
		print_line("Applied");
		print_line(next_state);
	}
	return next_state;
}

void PlannerPlan::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_verify_goals"), &PlannerPlan::get_verify_goals);
	ClassDB::bind_method(D_METHOD("set_verify_goals", "value"), &PlannerPlan::set_verify_goals);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "verify_goals"), "set_verify_goals", "get_verify_goals");

	ClassDB::bind_method(D_METHOD("get_verbose"), &PlannerPlan::get_verbose);
	ClassDB::bind_method(D_METHOD("set_verbose", "level"), &PlannerPlan::set_verbose);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "verbose"), "set_verbose", "get_verbose");

	ClassDB::bind_method(D_METHOD("get_domains"), &PlannerPlan::get_domains);
	ClassDB::bind_method(D_METHOD("set_domains", "domain"), &PlannerPlan::set_domains);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "domains", PROPERTY_HINT_RESOURCE_TYPE, "Domain"), "set_domains", "get_domains");

	ClassDB::bind_method(D_METHOD("get_current_domain"), &PlannerPlan::get_current_domain);
	ClassDB::bind_method(D_METHOD("set_current_domain", "current_domain"), &PlannerPlan::set_current_domain);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "current_domain", PROPERTY_HINT_RESOURCE_TYPE, "Domain"), "set_current_domain", "get_current_domain");

	ClassDB::bind_method(D_METHOD("find_plan", "state", "todo_list"), &PlannerPlan::find_plan);
	ClassDB::bind_method(D_METHOD("run_lazy_lookahead", "state", "todo_list", "max_tries"), &PlannerPlan::run_lazy_lookahead, DEFVAL(10));
	ClassDB::bind_method(D_METHOD("run_lazy_refineahead", "state", "todo_list"), &PlannerPlan::run_lazy_refineahead);
	ClassDB::bind_method(D_METHOD("generate_plan_id"), &PlannerPlan::generate_plan_id);
	ClassDB::bind_method(D_METHOD("submit_operation", "operation"), &PlannerPlan::submit_operation);
	ClassDB::bind_method(D_METHOD("get_global_state"), &PlannerPlan::get_global_state);
	
	// SQLite database methods
	ClassDB::bind_method(D_METHOD("initialize_database", "db_path"), &PlannerPlan::initialize_database, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("store_temporal_state", "state", "current_time"), &PlannerPlan::store_temporal_state);
	ClassDB::bind_method(D_METHOD("load_temporal_state"), &PlannerPlan::load_temporal_state);
	
	ADD_SIGNAL(MethodInfo("plan_id_generated", PropertyInfo(Variant::STRING, "plan_id")));
}

// Temporal method implementations
String PlannerPlan::generate_plan_id() {
    String uuid;
    Error err = CryptoCore::generate_uuidv7(uuid);
    if (err != OK) {
        ERR_PRINT("Failed to generate UUIDv7: " + itos(err));
        return String();
    }
    print_line("Generated plan ID: " + uuid);
    emit_signal("plan_id_generated", uuid);
    return uuid;
}

Dictionary PlannerPlan::submit_operation(Dictionary p_operation) {
    String transaction_id;
    Error err = CryptoCore::generate_uuidv7(transaction_id);
    if (err != OK) {
        ERR_PRINT("Failed to generate UUIDv7: " + itos(err));
        return Dictionary();
    }

    // Get absolute time in microseconds
    int64_t current_time = PlannerHLClock::now_microseconds();
    
    // Store operation in SQLite if database is initialized
    if (db.is_valid()) {
        store_planning_operation(transaction_id, "operation", p_operation, current_time);
    }

    Dictionary consensus_result;
    consensus_result["operation_id"] = transaction_id;
    consensus_result["agreed_at"] = current_time; // Absolute microseconds
    Array participants;
    participants.push_back("node_1");
    consensus_result["participants"] = participants;

    print_line("ParallelCommits operation submitted [" + transaction_id + "]: " + String(Variant(p_operation)));
    emit_signal("operation_submitted", consensus_result);
    return consensus_result;
}

Dictionary PlannerPlan::get_global_state() {
    // If database is initialized, load from SQLite
    if (db.is_valid()) {
        return load_temporal_state();
    }
    
    // Fallback to in-memory state if database not initialized
    Dictionary record;
    Array intent_writes;
    Dictionary tscache;

    Dictionary global_state;
    global_state["record"] = record;
    global_state["intent_writes"] = intent_writes;
    global_state["tscache"] = tscache;
    global_state["commit_ack"] = false;
    
    // Use current HLC from plan
    Dictionary hlc_dict;
    hlc_dict["l"] = hlc.get_start_time();
    hlc_dict["c"] = hlc.get_end_time();
    global_state["hlc"] = hlc_dict;

    return global_state;
}

bool PlannerPlan::get_verify_goals() const {
    return verify_goals;
}

void PlannerPlan::set_verify_goals(bool p_value) {
	verify_goals = p_value;
}

// SQLite database method implementations
bool PlannerPlan::initialize_database(const String &p_db_path) {
	db = memnew(SQLite);
	
	bool success = false;
	if (p_db_path.is_empty()) {
		// Use in-memory database
		success = db->open_in_memory();
	} else {
		// Use file-based database
		success = db->open(p_db_path);
	}
	
	if (!success) {
		ERR_PRINT("Failed to open SQLite database: " + db->get_last_error_message());
		db = Ref<SQLite>();
		return false;
	}
	
	// Create schema
	String create_temporal_state = "CREATE TABLE IF NOT EXISTS temporal_state ("
		"current_time INTEGER NOT NULL, "
		"timeline TEXT, "
		"last_updated INTEGER NOT NULL"
		");";
	
	String create_entity_capabilities = "CREATE TABLE IF NOT EXISTS entity_capabilities ("
		"entity_id TEXT NOT NULL, "
		"capability_name TEXT NOT NULL, "
		"capability_value TEXT, "
		"created_at INTEGER NOT NULL, "
		"updated_at INTEGER NOT NULL, "
		"PRIMARY KEY (entity_id, capability_name)"
		");";
	
	String create_planning_operations = "CREATE TABLE IF NOT EXISTS planning_operations ("
		"operation_id TEXT PRIMARY KEY, "
		"operation_type TEXT NOT NULL, "
		"operation_data TEXT, "
		"submitted_at INTEGER NOT NULL, "
		"status TEXT"
		");";
	
	String create_plan_history = "CREATE TABLE IF NOT EXISTS plan_history ("
		"plan_id TEXT PRIMARY KEY, "
		"state_snapshot TEXT, "
		"created_at INTEGER NOT NULL"
		");";
	
	Ref<SQLiteQuery> query1 = db->create_query(create_temporal_state);
	if (query1.is_null()) {
		ERR_PRINT("Failed to create temporal_state table");
		return false;
	}
	query1->execute(Array());
	
	Ref<SQLiteQuery> query2 = db->create_query(create_entity_capabilities);
	if (query2.is_null()) {
		ERR_PRINT("Failed to create entity_capabilities table");
		return false;
	}
	query2->execute(Array());
	
	Ref<SQLiteQuery> query3 = db->create_query(create_planning_operations);
	if (query3.is_null()) {
		ERR_PRINT("Failed to create planning_operations table");
		return false;
	}
	query3->execute(Array());
	
	Ref<SQLiteQuery> query4 = db->create_query(create_plan_history);
	if (query4.is_null()) {
		ERR_PRINT("Failed to create plan_history table");
		return false;
	}
	query4->execute(Array());
	
	print_line("SQLite database initialized successfully");
	return true;
}

void PlannerPlan::store_temporal_state(Dictionary p_state, int64_t p_current_time) {
	if (!db.is_valid()) {
		ERR_PRINT("Database not initialized");
		return;
	}
	
	String timeline_json = JSON::stringify(p_state);
	int64_t now = PlannerHLClock::now_microseconds();
	
	// Delete existing state
	Ref<SQLiteQuery> delete_query = db->create_query("DELETE FROM temporal_state");
	if (delete_query.is_valid()) {
		delete_query->execute(Array());
	}
	
	// Insert new state
	Ref<SQLiteQuery> insert_query = db->create_query(
		"INSERT INTO temporal_state (current_time, timeline, last_updated) VALUES (?, ?, ?)"
	);
	if (insert_query.is_valid()) {
		Array args;
		args.push_back(p_current_time);
		args.push_back(timeline_json);
		args.push_back(now);
		insert_query->execute(args);
	}
}

Dictionary PlannerPlan::load_temporal_state() {
	if (!db.is_valid()) {
		ERR_PRINT("Database not initialized");
		return Dictionary();
	}
	
	Ref<SQLiteQuery> query = db->create_query(
		"SELECT current_time, timeline, last_updated FROM temporal_state ORDER BY last_updated DESC LIMIT 1"
	);
	if (query.is_null()) {
		return Dictionary();
	}
	
	Variant result = query->execute(Array());
	if (result.get_type() != Variant::ARRAY) {
		return Dictionary();
	}
	
	Array rows = result;
	if (rows.is_empty()) {
		return Dictionary();
	}
	
	// SQLiteQuery returns Array of Arrays (rows), where each row is an Array of column values
	Array row = rows[0];
	if (row.size() < 3) {
		return Dictionary();
	}
	
	int64_t current_time = row[0]; // First column: current_time
	String timeline_json = row[1];  // Second column: timeline
	int64_t last_updated = row[2];  // Third column: last_updated
	
	Dictionary global_state;
	
	// Parse timeline JSON
	if (!timeline_json.is_empty()) {
		JSON json;
		Error err = json.parse(timeline_json);
		if (err == OK) {
			Variant parsed = json.get_data();
			if (parsed.get_type() == Variant::DICTIONARY) {
				global_state = parsed;
			}
		}
	}
	
	// Add HLC information
	Dictionary hlc_dict;
	hlc_dict["l"] = current_time;
	hlc_dict["c"] = current_time;
	global_state["hlc"] = hlc_dict;
	global_state["current_time"] = current_time;
	global_state["last_updated"] = last_updated;
	
	return global_state;
}

void PlannerPlan::store_entity_capability(const String &p_entity_id, const String &p_capability, Variant p_value, int64_t p_timestamp) {
	if (!db.is_valid()) {
		ERR_PRINT("Database not initialized");
		return;
	}
	
	String value_json = JSON::stringify(p_value);
	int64_t now = PlannerHLClock::now_microseconds();
	
	Ref<SQLiteQuery> query = db->create_query(
		"INSERT OR REPLACE INTO entity_capabilities (entity_id, capability_name, capability_value, created_at, updated_at) "
		"VALUES (?, ?, ?, COALESCE((SELECT created_at FROM entity_capabilities WHERE entity_id = ? AND capability_name = ?), ?), ?)"
	);
	if (query.is_valid()) {
		Array args;
		args.push_back(p_entity_id);
		args.push_back(p_capability);
		args.push_back(value_json);
		args.push_back(p_entity_id);
		args.push_back(p_capability);
		args.push_back(p_timestamp);
		args.push_back(now);
		query->execute(args);
	}
}

void PlannerPlan::store_planning_operation(const String &p_operation_id, const String &p_operation_type, Dictionary p_operation_data, int64_t p_timestamp) {
	if (!db.is_valid()) {
		ERR_PRINT("Database not initialized");
		return;
	}
	
	String operation_json = JSON::stringify(p_operation_data);
	
	Ref<SQLiteQuery> query = db->create_query(
		"INSERT INTO planning_operations (operation_id, operation_type, operation_data, submitted_at, status) "
		"VALUES (?, ?, ?, ?, ?)"
	);
	if (query.is_valid()) {
		Array args;
		args.push_back(p_operation_id);
		args.push_back(p_operation_type);
		args.push_back(operation_json);
		args.push_back(p_timestamp);
		args.push_back("submitted");
		query->execute(args);
	}
}

// Graph-based lazy refinement (Elixir-style)
Dictionary PlannerPlan::run_lazy_refineahead(Dictionary p_state, Array p_todo_list) {
	if (verbose >= 1) {
		print_line("run_lazy_refineahead: Starting graph-based planning");
		print_line("Initial state keys: " + String(Variant(p_state.keys())));
		print_line("Todo list: " + _item_to_string(p_todo_list));
	}
	
	// Initialize solution graph
	solution_graph = PlannerSolutionGraph();
	blacklisted_commands.clear();
	
	// Initialize STN solver
	stn.clear();
	stn.add_time_point("origin"); // Origin time point (plan start)
	
	// Initialize HLC if not already set
	if (hlc.get_start_time() == 0) {
		hlc.set_start_time(PlannerHLClock::now_microseconds());
	}
	
	// Anchor origin to current absolute time
	PlannerSTNConstraints::anchor_to_origin(stn, "origin", hlc.get_start_time());
	
	// Add initial tasks to the solution graph
	int parent_node_id = 0; // Root node
	PlannerGraphOperations::add_nodes_and_edges(
		solution_graph,
		parent_node_id,
		p_todo_list,
		current_domain->action_dictionary,
		current_domain->task_method_dictionary,
		current_domain->unigoal_method_dictionary,
		current_domain->multigoal_method_list
	);
	
	// Start planning loop
	Dictionary final_state = _planning_loop_recursive(parent_node_id, p_state, 0);
	
	// Update HLC with end time
	hlc.set_end_time(PlannerHLClock::now_microseconds());
	hlc.calculate_duration();
	
	// Store temporal state if database is initialized
	if (db.is_valid()) {
		store_temporal_state(final_state, hlc.get_end_time());
	}
	
	if (verbose >= 1) {
		print_line("run_lazy_refineahead: Completed graph-based planning");
		print_line("Duration: " + itos(hlc.get_duration()) + " microseconds");
	}
	
	return final_state;
}

Dictionary PlannerPlan::_planning_loop_recursive(int p_parent_node_id, Dictionary p_state, int p_iter) {
	if (verbose >= 2) {
		print_line(vformat("_planning_loop_recursive: parent_node_id=%d, iter=%d", p_parent_node_id, p_iter));
	}
	
	// Find the first Open node
	Variant open_node_result = PlannerGraphOperations::find_open_node(solution_graph, p_parent_node_id);
	
	if (open_node_result.get_type() == Variant::NIL) {
		// No open node found, check if parent is root
		Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
		int parent_type = parent_node["type"];
		
		if (parent_type == static_cast<int>(PlannerNodeType::TYPE_ROOT)) {
			// Planning complete
			if (verbose >= 1) {
				print_line("Planning complete, returning final state");
			}
			return p_state;
		} else {
			// Move to predecessor
			int new_parent = PlannerGraphOperations::find_predecessor(solution_graph, p_parent_node_id);
			if (new_parent >= 0) {
				return _planning_loop_recursive(new_parent, p_state, p_iter + 1);
			}
			return p_state;
		}
	}
	
	int curr_node_id = open_node_result;
	Dictionary curr_node = solution_graph.get_node(curr_node_id);
	
	if (verbose >= 2) {
		print_line(vformat("Iteration %d: Refining node %d", p_iter, curr_node_id));
	}
	
	// Save current state if first visit (state is empty)
	Dictionary node_state = solution_graph.get_state_snapshot(curr_node_id);
	if (node_state.is_empty()) {
		solution_graph.save_state_snapshot(curr_node_id, p_state.duplicate());
		// Also save STN snapshot on first visit
		PlannerSTNSolver::Snapshot snapshot = stn.create_snapshot();
		curr_node["stn_snapshot"] = snapshot.to_dictionary();
		solution_graph.update_node(curr_node_id, curr_node);
	} else {
		// Restore state if backtracking
		p_state = node_state.duplicate();
		// Also restore STN snapshot
		_restore_stn_from_node(curr_node_id);
	}
	
	int node_type = curr_node["type"];
	
	// Handle different node types
	switch (static_cast<PlannerNodeType>(node_type)) {
		case PlannerNodeType::TYPE_TASK: {
			// Try to refine task with available methods (like Elixir's Enum.find_value)
			Variant task_info = curr_node["info"];
			
			// Extract metadata and validate entity requirements
			PlannerMetadata metadata = _extract_metadata(task_info);
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("Task entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			
			TypedArray<Callable> available_methods = curr_node["available_methods"];
			
			// Try all available methods (like Elixir's Enum.find_value)
			// Don't modify available_methods - keep full list for backtracking
			Callable selected_method;
			Array subtasks;
			bool found_working_method = false;
			
			for (int i = 0; i < available_methods.size(); i++) {
				Callable method = available_methods[i];
				Array task_arr = task_info;
				Array args;
				args.push_back(p_state);
				args.append_array(task_arr.slice(1));
				
				Variant result = method.callv(args);
				if (result.get_type() == Variant::ARRAY) {
					subtasks = result;
					selected_method = method;
					found_working_method = true;
					break; // Found working method, stop trying
				}
				// Method failed, continue to next (like Enum.find_value)
			}
			
			if (found_working_method) {
				// Successfully refined - like Elixir's {method, subtasks}
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["selected_method"] = selected_method;
				// Don't modify available_methods - keep full list for potential backtracking
				solution_graph.update_node(curr_node_id, curr_node);
				
				// Add subtasks to graph
				PlannerGraphOperations::add_nodes_and_edges(
					solution_graph,
					curr_node_id,
					subtasks,
					current_domain->action_dictionary,
					current_domain->task_method_dictionary,
					current_domain->unigoal_method_dictionary,
					current_domain->multigoal_method_list
				);
				
				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}
			
			// Failed to refine, backtrack
			if (verbose >= 2) {
				print_line("Task refinement failed, backtracking");
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
				solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
			);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}
		
		case PlannerNodeType::TYPE_ACTION: {
			Variant action_info = curr_node["info"];
			
			// Check if blacklisted
			if (_is_command_blacklisted(action_info)) {
				if (verbose >= 2) {
					print_line("Action is blacklisted, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			
			// Create STN snapshot before action execution and store with node
			stn_snapshot = stn.create_snapshot();
			curr_node["stn_snapshot"] = stn_snapshot.to_dictionary();
			solution_graph.update_node(curr_node_id, curr_node);
			
			// Check for temporal constraints and entity requirements in action
			PlannerMetadata metadata = _extract_metadata(action_info);
			Dictionary temporal_metadata;
			if (_has_temporal_constraints(action_info)) {
				temporal_metadata = _get_temporal_constraints(action_info);
			}
			
			// Validate entity requirements before executing action
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("Action entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			
			// Execute action with temporal tracking
			Callable action = curr_node["action"];
			Array action_arr = action_info;
			Array args;
			args.push_back(p_state);
			args.append_array(action_arr.slice(1));
			
			int64_t action_start_time = PlannerHLClock::now_microseconds();
			
			// Use temporal metadata start_time if provided
			if (temporal_metadata.has("start_time")) {
				String start_time_str = temporal_metadata["start_time"];
				// Convert ISO 8601 to microseconds (simplified - would need proper parsing)
				// For now, use actual time if metadata start_time is in the future
			}
			
			Variant result = action.callv(args);
			int64_t action_end_time = PlannerHLClock::now_microseconds();
			int64_t action_duration = action_end_time - action_start_time;
			
			// Use temporal metadata duration if provided
			if (temporal_metadata.has("duration")) {
				String duration_str = temporal_metadata["duration"];
				// Convert ISO 8601 duration to microseconds (simplified)
				// For now, use actual measured duration
			}
			
			if (result.get_type() == Variant::DICTIONARY) {
				Dictionary new_state = result;
				
				// Add action interval to STN
				String action_id = action_arr[0];
				int64_t metadata_start = action_start_time;
				int64_t metadata_end = action_end_time;
				
				// Use temporal metadata times if provided
				if (temporal_metadata.has("start_time")) {
					// Parse ISO 8601 start_time to microseconds (simplified for now)
					// For now, use actual start_time
				}
				if (temporal_metadata.has("end_time")) {
					// Parse ISO 8601 end_time to microseconds (simplified for now)
					// For now, use actual end_time
				}
				
				/*bool stn_success =*/ PlannerSTNConstraints::add_interval(
					stn, action_id, metadata_start, metadata_end, action_duration
				);
				
				// Check STN consistency
				stn.check_consistency();
				if (!stn.is_consistent()) {
					// STN inconsistent, backtrack
					if (verbose >= 2) {
						print_line("STN inconsistent after action, backtracking");
					}
					_blacklist_command(action_info);
					PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
					);
					solution_graph = backtrack_result.graph;
					if (backtrack_result.parent_node_id >= 0) {
						// Restore STN snapshot from the node we're backtracking to
						_restore_stn_from_node(backtrack_result.parent_node_id);
						return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
					}
					return p_state;
				}
				
				// Action successful and STN consistent
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["start_time"] = action_start_time;
				curr_node["end_time"] = action_end_time;
				curr_node["duration"] = action_duration;
				solution_graph.update_node(curr_node_id, curr_node);
				
				// Update plan HLC
				hlc.set_end_time(action_end_time);
				hlc.calculate_duration();
				
				return _planning_loop_recursive(p_parent_node_id, new_state, p_iter + 1);
			} else {
				// Action failed, backtrack and restore STN
				if (verbose >= 2) {
					print_line("Action execution failed, backtracking");
				}
				_blacklist_command(action_info);
				stn.restore_snapshot(stn_snapshot);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
		}
		
		case PlannerNodeType::TYPE_GOAL: {
			Array goal_arr = curr_node["info"];
			if (goal_arr.size() < 3) {
				// Invalid goal format
				return p_state;
			}
			
			String state_var_name = goal_arr[0];
			String argument = goal_arr[1];
			Variant desired_value = goal_arr[2];
			
			// Extract metadata and validate entity requirements
			PlannerMetadata metadata = _extract_metadata(goal_arr);
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("Goal entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			
			// Check if goal already achieved
			Dictionary state_var = p_state[state_var_name];
			if (state_var[argument] == desired_value) {
				// Goal already achieved
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(curr_node_id, curr_node);
				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}
			
			// Try to refine goal (like Elixir's Enum.find_value)
			TypedArray<Callable> available_methods = curr_node["available_methods"];
			
			// Try all available methods - don't modify available_methods
			Callable selected_method;
			Array subgoals;
			bool found_working_method = false;
			
			for (int i = 0; i < available_methods.size(); i++) {
				Callable method = available_methods[i];
				Variant result = method.call(p_state, argument, desired_value);
				if (result.get_type() == Variant::ARRAY) {
					subgoals = result;
					selected_method = method;
					found_working_method = true;
					break;
				}
				// Method failed, continue to next
			}
			
			if (found_working_method) {
				// Successfully refined
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["selected_method"] = selected_method;
				// Don't modify available_methods
				solution_graph.update_node(curr_node_id, curr_node);
				
				// Add subgoals to graph
				PlannerGraphOperations::add_nodes_and_edges(
					solution_graph,
					curr_node_id,
					subgoals,
					current_domain->action_dictionary,
					current_domain->task_method_dictionary,
					current_domain->unigoal_method_dictionary,
					current_domain->multigoal_method_list
				);
				
				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}
			
			// Failed to refine, backtrack
			if (verbose >= 2) {
				print_line("Goal refinement failed, backtracking");
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
				solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
			);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}
		
		case PlannerNodeType::TYPE_MULTIGOAL: {
			Ref<PlannerMultigoal> multigoal = curr_node["info"];
			if (multigoal.is_null()) {
				return p_state;
			}
			
			// Extract metadata from multigoal and validate entity requirements
			// Multigoal metadata might be stored in the multigoal object itself
			Variant multigoal_variant = multigoal;
			PlannerMetadata metadata = _extract_metadata(multigoal_variant);
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("MultiGoal entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			
			// Check if multigoal already achieved
			Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, multigoal);
			if (goals_not_achieved.is_empty()) {
				// All goals are already achieved
				if (verbose >= 1) {
					print_line("MultiGoal already achieved, marking as closed");
				}
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(curr_node_id, curr_node);
				// Add empty subgoals for verification node (like Elixir)
				Array empty_subgoals;
				PlannerGraphOperations::add_nodes_and_edges(
					solution_graph,
					curr_node_id,
					empty_subgoals,
					current_domain->action_dictionary,
					current_domain->task_method_dictionary,
					current_domain->unigoal_method_dictionary,
					current_domain->multigoal_method_list
				);
				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}
			
			// Try to refine multigoal (like Elixir's Enum.find_value)
			TypedArray<Callable> available_methods = curr_node["available_methods"];
			
			// Try all available methods - don't modify available_methods
			Callable selected_method;
			Array subgoals;
			bool found_working_method = false;
			
			for (int i = 0; i < available_methods.size(); i++) {
				Callable method = available_methods[i];
				Variant result = method.call(p_state, multigoal);
				if (result.get_type() == Variant::ARRAY) {
					subgoals = result;
					selected_method = method;
					found_working_method = true;
					break;
				}
				// Method failed, continue to next
			}
			
			if (found_working_method) {
				// Successfully refined
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["selected_method"] = selected_method;
				// Don't modify available_methods
				solution_graph.update_node(curr_node_id, curr_node);
				
				// Optimize unigoal order (most constraining first) before adding to graph
				Array optimized_subgoals = _optimize_unigoal_order(
					subgoals, p_state, current_domain->unigoal_method_dictionary
				);
				
				// Add optimized subgoals to graph
				PlannerGraphOperations::add_nodes_and_edges(
					solution_graph,
					curr_node_id,
					optimized_subgoals,
					current_domain->action_dictionary,
					current_domain->task_method_dictionary,
					current_domain->unigoal_method_dictionary,
					current_domain->multigoal_method_list
				);
				
				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}
			
			// Failed to refine, backtrack
			if (verbose >= 2) {
				print_line("MultiGoal refinement failed, backtracking");
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
				solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
			);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}
		
		case PlannerNodeType::TYPE_VERIFY_GOAL: {
			// Verify the parent goal
			Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
			Array goal_arr = parent_node["info"];
			if (goal_arr.size() >= 3) {
				String state_var_name = goal_arr[0];
				String argument = goal_arr[1];
				Variant desired_value = goal_arr[2];
				
				Dictionary state_var = p_state[state_var_name];
				if (state_var[argument] == desired_value) {
					// Verification successful
					curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
					solution_graph.update_node(curr_node_id, curr_node);
					return _planning_loop_recursive(p_parent_node_id, p_state, p_iter + 1);
				}
			}
			
			// Verification failed, backtrack
			if (verbose >= 2) {
				print_line("Goal verification failed, backtracking");
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
				solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
			);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}
		
		case PlannerNodeType::TYPE_VERIFY_MULTIGOAL: {
			// Verify the parent multigoal
			Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
			Ref<PlannerMultigoal> multigoal = parent_node["info"];
			if (multigoal.is_null()) {
				// Invalid parent, backtrack
				if (verbose >= 2) {
					print_line("MultiGoal verification failed: invalid parent multigoal, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			
			Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, multigoal);
			if (goals_not_achieved.is_empty()) {
				// Verification successful - all goals are achieved
				if (verbose >= 1) {
					print_line("MultiGoal verified successfully");
				}
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(curr_node_id, curr_node);
				return _planning_loop_recursive(p_parent_node_id, p_state, p_iter + 1);
			} else {
				// Verification failed - some goals not achieved
				if (verbose >= 2) {
					print_line("MultiGoal verification failed: some goals not achieved, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands
				);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
		}
		
		default:
			return p_state;
	}
}

void PlannerPlan::_restore_stn_from_node(int p_node_id) {
	if (p_node_id >= 0) {
		Dictionary node = solution_graph.get_node(p_node_id);
		if (node.has("stn_snapshot")) {
			Dictionary snapshot_dict = node["stn_snapshot"];
			PlannerSTNSolver::Snapshot snapshot = PlannerSTNSolver::Snapshot::from_dictionary(snapshot_dict);
			stn.restore_snapshot(snapshot);
			if (verbose >= 3) {
				print_line("Restored STN snapshot from node " + itos(p_node_id));
			}
		}
	}
}

bool PlannerPlan::_is_command_blacklisted(Variant p_command) const {
	// Compare Arrays properly - need to check if it's an Array and compare elements
	if (p_command.get_type() != Variant::ARRAY) {
		return false;
	}
	
	Array action_arr = p_command;
	
	// Check each blacklisted command
	for (int i = 0; i < blacklisted_commands.size(); i++) {
		Variant blacklisted = blacklisted_commands[i];
		if (blacklisted.get_type() != Variant::ARRAY) {
			continue;
		}
		
		Array blacklisted_arr = blacklisted;
		
		// Compare Arrays element by element
		if (blacklisted_arr.size() != action_arr.size()) {
			continue;
		}
		
		bool match = true;
		for (int j = 0; j < action_arr.size(); j++) {
			if (action_arr[j] != blacklisted_arr[j]) {
				match = false;
				break;
			}
		}
		
		if (match) {
			return true;
		}
	}
	return false;
}

void PlannerPlan::_blacklist_command(Variant p_command) {
	if (!_is_command_blacklisted(p_command)) {
		blacklisted_commands.push_back(p_command);
	}
}

// Goal solver methods (moved from PlannerGoalSolver)

PlannerPlan::ConstrainingFactor PlannerPlan::_calculate_constraining_factor(const Variant &p_goal, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict) const {
	ConstrainingFactor factor;
	
	// Extract goal info (assuming format: [state_var_name, argument, desired_value])
	if (p_goal.get_type() != Variant::ARRAY) {
		return factor;
	}
	
	Array goal_arr = p_goal;
	if (goal_arr.size() < 3) {
		return factor;
	}
	
	String state_var_name = goal_arr[0];
	String argument = goal_arr[1];
	Variant value = goal_arr[2];
	
	// Count available methods for this unigoal (optimization strategy 1: total method count)
	if (p_unigoal_method_dict.has(state_var_name)) {
		Variant methods_var = p_unigoal_method_dict[state_var_name];
		if (methods_var.get_type() == Variant::ARRAY) {
			TypedArray<Callable> methods = methods_var;
			factor.total_method_count = methods.size();
			
			// Optimization strategy 2: count only applicable methods in current state
			// Try each method to see if it's applicable (returns Array if applicable, false otherwise)
			for (int i = 0; i < methods.size(); i++) {
				Callable method = methods[i];
				Variant result = method.call(p_state, argument, value);
				if (result.get_type() == Variant::ARRAY) {
					// Method is applicable in current state
					factor.applicable_method_count++;
				}
			}
		}
	}
	
	// Check for temporal constraints
	PlannerMetadata metadata = _extract_temporal_constraints(p_goal);
	if (!metadata.start_time.is_empty() || !metadata.end_time.is_empty()) {
		factor.has_temporal_constraints = true;
	}
	
	return factor;
}

PlannerMetadata PlannerPlan::_extract_temporal_constraints(const Variant &p_item) const {
	// Extract only temporal constraints (for backward compatibility)
	PlannerMetadata metadata = _extract_metadata(p_item);
	// Clear entity requirements to return only temporal constraints
	metadata.requires_entities.clear();
	return metadata;
}

PlannerMetadata PlannerPlan::_extract_metadata(const Variant &p_item) const {
	PlannerMetadata metadata;
	
	// Check if item has temporal_constraints or metadata field
	if (p_item.get_type() == Variant::DICTIONARY) {
		Dictionary item_dict = p_item;
		if (item_dict.has("temporal_constraints")) {
			Dictionary temporal_dict = item_dict["temporal_constraints"];
			metadata = PlannerMetadata::from_dictionary(temporal_dict);
		} else if (item_dict.has("metadata")) {
			Dictionary metadata_dict = item_dict["metadata"];
			metadata = PlannerMetadata::from_dictionary(metadata_dict);
		}
	} else if (p_item.get_type() == Variant::ARRAY) {
		Array item_arr = p_item;
		// Metadata might be stored as last element or in a wrapper
		// Check if last element is a dictionary with temporal_constraints or metadata
		if (item_arr.size() > 0) {
			Variant last = item_arr[item_arr.size() - 1];
			if (last.get_type() == Variant::DICTIONARY) {
				Dictionary last_dict = last;
				if (last_dict.has("temporal_constraints")) {
					Dictionary temporal_dict = last_dict["temporal_constraints"];
					metadata = PlannerMetadata::from_dictionary(temporal_dict);
				} else if (last_dict.has("metadata")) {
					Dictionary metadata_dict = last_dict["metadata"];
					metadata = PlannerMetadata::from_dictionary(metadata_dict);
				}
			}
		}
	}
	
	return metadata;
}

Array PlannerPlan::_optimize_unigoal_order(const Array &p_unigoals, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict) {
	// Use LocalVector internally for efficiency
	LocalVector<GoalWithFactor> goals_with_factors;
	
	// Calculate constraining factors for each unigoal
	for (int i = 0; i < p_unigoals.size(); i++) {
		Variant goal = p_unigoals[i];
		ConstrainingFactor factor = _calculate_constraining_factor(goal, p_state, p_unigoal_method_dict);
		goals_with_factors.push_back(GoalWithFactor(goal, factor));
	}
	
	// Sort by constraining factor (most constraining first)
	// Use insertion sort for small arrays, or std::sort-like approach
	if (goals_with_factors.size() > 1) {
		// Simple insertion sort: most constraining first
		for (uint32_t i = 1; i < goals_with_factors.size(); i++) {
			GoalWithFactor key = goals_with_factors[i];
			int j = i - 1;
			
			// Move elements with less constraining factors to the right
			while (j >= 0 && goals_with_factors[j].factor < key.factor) {
				goals_with_factors[j + 1] = goals_with_factors[j];
				j--;
			}
			goals_with_factors[j + 1] = key;
		}
	}
	
	// Convert back to Array for GDScript interface
	Array ordered_goals;
	ordered_goals.resize(goals_with_factors.size());
	for (uint32_t i = 0; i < goals_with_factors.size(); i++) {
		ordered_goals[i] = goals_with_factors[i].goal;
	}
	
	return ordered_goals;
}

Variant PlannerPlan::_attach_temporal_constraints(const Variant &p_item, const Dictionary &p_temporal_constraints) {
	PlannerMetadata metadata = PlannerMetadata::from_dictionary(p_temporal_constraints);
	
	// Create a wrapper dictionary with the item and temporal constraints
	Dictionary result;
	
	if (p_item.get_type() == Variant::DICTIONARY) {
		// If already a dictionary, add temporal_constraints field
		result = Dictionary(p_item);
		result["temporal_constraints"] = metadata.to_dictionary();
	} else if (p_item.get_type() == Variant::ARRAY) {
		// If array, wrap in dictionary with temporal constraints
		result["item"] = p_item;
		result["temporal_constraints"] = metadata.to_dictionary();
	} else {
		// For other types, wrap in dictionary
		result["item"] = p_item;
		result["temporal_constraints"] = metadata.to_dictionary();
	}
	
	return result;
}

Dictionary PlannerPlan::_get_temporal_constraints(const Variant &p_item) const {
	PlannerMetadata metadata = _extract_temporal_constraints(p_item);
	return metadata.to_dictionary();
}

bool PlannerPlan::_has_temporal_constraints(const Variant &p_item) const {
	PlannerMetadata metadata = _extract_temporal_constraints(p_item);
	return !metadata.start_time.is_empty() || !metadata.end_time.is_empty() || !metadata.duration.is_empty();
}

bool PlannerPlan::_validate_entity_requirements(const Dictionary &p_state, const PlannerMetadata &p_metadata) const {
	// Check if metadata has entity requirements
	if (p_metadata.requires_entities.size() == 0) {
		return true; // No entity requirements, validation passes
	}
	
	// Match entities for all requirements
	Dictionary match_result = _match_entities(p_state, p_metadata.requires_entities);
	bool success = match_result["success"];
	
	if (!success && verbose >= 2) {
		String error = match_result["error"];
		print_line("Entity matching failed: " + error);
	}
	
	return success;
}

Dictionary PlannerPlan::_match_entities(const Dictionary &p_state, const LocalVector<PlannerEntityRequirement> &p_requirements) const {
	Dictionary result;
	result["success"] = false;
	result["matched_entities"] = Array();
	result["error"] = "";
	
	// Use internal HashMap/LocalVector for efficiency
	HashMap<String, String> entity_types; // entity_id -> type
	HashMap<String, LocalVector<String>> entity_capabilities; // entity_id -> capabilities
	
	// Extract entity data from state
	// State structure: entities are stored in a nested structure
	// We'll look for entity_capabilities or similar structure
	if (p_state.has("entity_capabilities")) {
		Dictionary entity_caps_dict = p_state["entity_capabilities"];
		Array entity_ids = entity_caps_dict.keys();
		
		for (int i = 0; i < entity_ids.size(); i++) {
			String entity_id = entity_ids[i];
			Dictionary entity_data = entity_caps_dict[entity_id];
			
			// Extract type (stored as "type" capability)
			if (entity_data.has("type")) {
				entity_types[entity_id] = entity_data["type"];
			}
			
			// Extract all capabilities (any non-type key that has a truthy value)
			LocalVector<String> caps;
			Array cap_keys = entity_data.keys();
			for (int j = 0; j < cap_keys.size(); j++) {
				String cap_key = cap_keys[j];
				if (cap_key != "type") {
					Variant cap_value = entity_data[cap_key];
					// Include capability if value is truthy (true, non-zero, non-empty)
					if (cap_value.operator bool()) {
						caps.push_back(cap_key);
					}
				}
			}
			entity_capabilities[entity_id] = caps;
		}
	}
	
	// Match entities to requirements
	Array matched_entities;
	
	// Match each requirement to an entity
	for (uint32_t req_idx = 0; req_idx < p_requirements.size(); req_idx++) {
		const PlannerEntityRequirement &req = p_requirements[req_idx];
		bool matched = false;
		
		// Try to find matching entity
		for (const KeyValue<String, String> &E : entity_types) {
			String entity_id = E.key;
			String entity_type = E.value;
			
			// Check type match
			if (entity_type != req.type) {
				continue;
			}
			
			// Check if entity has all required capabilities
			const LocalVector<String> *entity_caps = entity_capabilities.getptr(entity_id);
			if (entity_caps == nullptr) {
				continue;
			}
			
			bool has_all_caps = true;
			for (uint32_t cap_idx = 0; cap_idx < req.capabilities.size(); cap_idx++) {
				String required_cap = req.capabilities[cap_idx];
				bool found = false;
				for (uint32_t j = 0; j < entity_caps->size(); j++) {
					if ((*entity_caps)[j] == required_cap) {
						found = true;
						break;
					}
				}
				if (!found) {
					has_all_caps = false;
					break;
				}
			}
			
			if (has_all_caps) {
				// Found matching entity
				matched_entities.push_back(entity_id);
				matched = true;
				break;
			}
		}
		
		if (!matched) {
			result["success"] = false;
			// Convert capabilities to string for error message
			String caps_str = "[";
			for (uint32_t i = 0; i < req.capabilities.size(); i++) {
				if (i > 0) caps_str += ", ";
				caps_str += req.capabilities[i];
			}
			caps_str += "]";
			result["error"] = vformat("No entity found matching requirement: type=%s, capabilities=%s", req.type, caps_str);
			return result;
		}
	}
	
	result["success"] = true;
	result["matched_entities"] = matched_entities;
	return result;
}
