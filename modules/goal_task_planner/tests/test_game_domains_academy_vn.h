/**************************************************************************/
/*  test_game_domains_academy_vn.h                                        */
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

// Academy Visual Novel domain for game-inspired planner tests

#pragma once

#include "../domain.h"
#include "../multigoal.h"
#include "test_game_domains_helpers.h"

#ifdef TOOLS_ENABLED
namespace TestGameDomainsBacktracking {

// ============================================================================
// ACADEMY VISUAL NOVEL DOMAIN
// ============================================================================

// Actions
static Variant action_talk_to_character(Dictionary p_state, String p_character_id, int64_t p_timeslot) {
	Dictionary characters = p_state["characters"];
	Dictionary schedule = p_state["schedule"];
	Dictionary relationships = p_state["relationships"];

	// Check if character exists and is available at this timeslot
	if (!characters.has(p_character_id)) {
		return false;
	}
	Dictionary character = characters[p_character_id];

	// Check if character is available at this timeslot
	Dictionary char_schedule;
	if (character.has("schedule")) {
		char_schedule = character["schedule"];
	}
	if (char_schedule.has(p_timeslot) && char_schedule[p_timeslot].operator bool()) {
		return false; // Character is busy
	}

	// Check if player is available at this timeslot
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}
	if (player_schedule.has(p_timeslot) && player_schedule[p_timeslot].operator bool()) {
		return false; // Player is busy
	}

	// Talk to character (increases relationship)
	Dictionary new_state = p_state.duplicate();
	Dictionary new_relationships = relationships.duplicate();
	Dictionary new_schedule = schedule.duplicate();
	Dictionary new_player_schedule = player_schedule.duplicate();

	// Update relationship
	int current_level = new_relationships.has(p_character_id) ? new_relationships[p_character_id].operator int() : 0;
	new_relationships[p_character_id] = current_level + 1;

	// Mark timeslot as used
	new_player_schedule[p_timeslot] = true;
	new_schedule["player"] = new_player_schedule;

	new_state["relationships"] = new_relationships;
	new_state["schedule"] = new_schedule;
	return new_state;
}

static Variant action_attend_class(Dictionary p_state, String p_class_id, int64_t p_timeslot) {
	Dictionary classes = p_state["classes"];
	Dictionary schedule = p_state["schedule"];
	Dictionary enrolled = p_state["enrolled"];

	// Check if class exists
	if (!classes.has(p_class_id)) {
		return false;
	}
	Dictionary class_info = classes[p_class_id];

	// Check if enrolled
	if (!enrolled.has(p_class_id) || !enrolled[p_class_id].operator bool()) {
		return false;
	}

	// Check if player is available at this timeslot
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}
	if (player_schedule.has(p_timeslot) && player_schedule[p_timeslot].operator bool()) {
		return false; // Player is busy
	}

	// Check if class is scheduled at this timeslot
	int64_t class_timeslot = class_info.has("timeslot") ? class_info["timeslot"].operator int64_t() : -1;
	if (class_timeslot != -1 && class_timeslot != p_timeslot) {
		return false; // Class is at different timeslot
	}

	// Attend class
	Dictionary new_state = p_state.duplicate();
	Dictionary new_schedule = schedule.duplicate();
	Dictionary new_player_schedule = player_schedule.duplicate();
	Dictionary new_attended;
	if (p_state.has("attended")) {
		new_attended = p_state["attended"].duplicate();
	}

	new_player_schedule[p_timeslot] = true;
	new_schedule["player"] = new_player_schedule;

	// Mark class as attended
	Array attended_list;
	if (new_attended.has(p_class_id)) {
		attended_list = new_attended[p_class_id];
	}
	attended_list.push_back(p_timeslot);
	new_attended[p_class_id] = attended_list;

	new_state["schedule"] = new_schedule;
	new_state["attended"] = new_attended;
	return new_state;
}

static Variant action_study(Dictionary p_state, String p_subject, int64_t p_timeslot) {
	Dictionary schedule = p_state["schedule"];

	// Check if player is available at this timeslot
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}
	if (player_schedule.has(p_timeslot) && player_schedule[p_timeslot].operator bool()) {
		return false; // Player is busy
	}

	// Study
	Dictionary new_state = p_state.duplicate();
	Dictionary new_schedule = schedule.duplicate();
	Dictionary new_player_schedule = player_schedule.duplicate();
	Dictionary new_study_progress;
	if (p_state.has("study_progress")) {
		new_study_progress = p_state["study_progress"].duplicate();
	}

	new_player_schedule[p_timeslot] = true;
	new_schedule["player"] = new_player_schedule;

	// Update study progress
	int current_progress = new_study_progress.has(p_subject) ? new_study_progress[p_subject].operator int() : 0;
	new_study_progress[p_subject] = current_progress + 1;

	new_state["schedule"] = new_schedule;
	new_state["study_progress"] = new_study_progress;
	return new_state;
}

static Variant action_participate_event(Dictionary p_state, String p_event_id, int64_t p_timeslot) {
	Dictionary events = p_state["events"];
	Dictionary schedule = p_state["schedule"];
	Dictionary story_flags = p_state["story_flags"];

	// Check if event exists
	if (!events.has(p_event_id)) {
		return false;
	}
	Dictionary event_info = events[p_event_id];

	// Check prerequisites
	if (event_info.has("prerequisites")) {
		Array prerequisites = event_info["prerequisites"];
		for (int i = 0; i < prerequisites.size(); i++) {
			String prereq_flag = prerequisites[i];
			if (!story_flags.has(prereq_flag) || !story_flags[prereq_flag].operator bool()) {
				return false; // Prerequisite not met
			}
		}
	}

	// Check if player is available at this timeslot
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}
	if (player_schedule.has(p_timeslot) && player_schedule[p_timeslot].operator bool()) {
		return false; // Player is busy
	}

	// Participate in event
	Dictionary new_state = p_state.duplicate();
	Dictionary new_schedule = schedule.duplicate();
	Dictionary new_player_schedule = player_schedule.duplicate();
	Dictionary new_story_flags = story_flags.duplicate();
	Dictionary new_completed_events;
	if (p_state.has("completed_events")) {
		new_completed_events = p_state["completed_events"].duplicate();
	}

	new_player_schedule[p_timeslot] = true;
	new_schedule["player"] = new_player_schedule;

	// Mark event as completed
	new_completed_events[p_event_id] = true;

	// Set story flags
	if (event_info.has("sets_flags")) {
		Array sets_flags = event_info["sets_flags"];
		for (int i = 0; i < sets_flags.size(); i++) {
			String flag = sets_flags[i];
			new_story_flags[flag] = true;
		}
	}

	new_state["schedule"] = new_schedule;
	new_state["story_flags"] = new_story_flags;
	new_state["completed_events"] = new_completed_events;
	return new_state;
}

static Variant action_make_choice(Dictionary p_state, String p_choice_id, Variant p_choice_value) {
	Dictionary choices = p_state["choices"];
	Dictionary story_flags = p_state["story_flags"];

	// Make choice (sets story flag)
	Dictionary new_state = p_state.duplicate();
	Dictionary new_story_flags = story_flags.duplicate();

	new_story_flags[p_choice_id] = p_choice_value;

	new_state["story_flags"] = new_story_flags;
	return new_state;
}

static Variant action_spend_time_with(Dictionary p_state, String p_character_id, int64_t p_timeslot) {
	// Similar to talk_to_character but with more relationship gain
	return action_talk_to_character(p_state, p_character_id, p_timeslot);
}

// Task methods
static Variant task_method_build_relationship(Dictionary p_state, String p_character_id, int p_target_level) {
	// Check if character exists
	Dictionary characters = p_state["characters"];
	if (!characters.has(p_character_id)) {
		return false; // Character doesn't exist
	}

	Dictionary relationships = p_state["relationships"];
	int current_level = relationships.has(p_character_id) ? relationships[p_character_id].operator int() : 0;

	if (current_level >= p_target_level) {
		return Array(); // Already at target level
	}

	// Need to spend time with character
	Array subtasks;
	Dictionary schedule = p_state["schedule"];
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}

	// Find available timeslot
	int64_t available_timeslot = -1;
	for (int64_t slot = 0; slot < 24; slot++) {
		if (!player_schedule.has(slot) || !player_schedule[slot].operator bool()) {
			// Check if character is also available at this slot
			Dictionary character = characters[p_character_id];
			Dictionary char_schedule;
			if (character.has("schedule")) {
				char_schedule = character["schedule"];
			}
			if (!char_schedule.has(slot) || !char_schedule[slot].operator bool()) {
				available_timeslot = slot;
				break;
			}
		}
	}

	if (available_timeslot == -1) {
		return false; // No available timeslot
	}

	Array action;
	action.push_back("action_spend_time_with");
	action.push_back(p_character_id);
	action.push_back(available_timeslot);
	subtasks.push_back(action);

	// If still need more relationship, continue building
	if (current_level + 1 < p_target_level) {
		Array task;
		task.push_back("build_relationship");
		task.push_back(p_character_id);
		task.push_back(p_target_level);
		subtasks.push_back(task);
	}

	return subtasks;
}

static Variant task_method_complete_story_route(Dictionary p_state, String p_route_id) {
	Dictionary routes = p_state["routes"];
	Dictionary story_flags = p_state["story_flags"];

	if (!routes.has(p_route_id)) {
		return false;
	}
	Dictionary route = routes[p_route_id];

	// Check route requirements
	Dictionary requirements;
	if (route.has("requirements")) {
		requirements = route["requirements"];
	}

	// Check if already completed
	if (story_flags.has("route_" + p_route_id + "_completed") && story_flags["route_" + p_route_id + "_completed"].operator bool()) {
		return Array(); // Already completed
	}

	Array subtasks;

	// Check relationship requirements
	if (requirements.has("relationships")) {
		Dictionary rel_requirements = requirements["relationships"];
		Array rel_keys = rel_requirements.keys();
		for (int i = 0; i < rel_keys.size(); i++) {
			String char_id = rel_keys[i];
			int required_level = rel_requirements[char_id].operator int();

			Array task;
			task.push_back("build_relationship");
			task.push_back(char_id);
			task.push_back(required_level);
			subtasks.push_back(task);
		}
	}

	// Check event requirements
	if (requirements.has("events")) {
		Array event_requirements = requirements["events"];
		for (int i = 0; i < event_requirements.size(); i++) {
			String event_id = event_requirements[i];

			Array task;
			task.push_back("trigger_event");
			task.push_back(event_id);
			subtasks.push_back(task);
		}
	}

	// Check story flag requirements
	if (requirements.has("story_flags")) {
		Array flag_requirements = requirements["story_flags"];
		for (int i = 0; i < flag_requirements.size(); i++) {
			String flag = flag_requirements[i];
			if (!story_flags.has(flag) || !story_flags[flag].operator bool()) {
				// Need to set this flag (simplified - would need to find how to set it)
				return false; // Cannot set flag automatically
			}
		}
	}

	if (subtasks.is_empty()) {
		return Array(); // All requirements met
	}
	return subtasks;
}

static Variant task_method_manage_time(Dictionary p_state, Dictionary p_activities) {
	// Activities: {timeslot: activity_type, ...}
	Dictionary schedule = p_state["schedule"];
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}

	Array subtasks;
	Array activity_keys = p_activities.keys();

	for (int i = 0; i < activity_keys.size(); i++) {
		int64_t timeslot = activity_keys[i].operator int64_t();
		Dictionary activity = p_activities[timeslot];
		String activity_type = activity["type"];

		// Check if timeslot is available
		if (player_schedule.has(timeslot) && player_schedule[timeslot].operator bool()) {
			// Conflict - need to resolve
			return false; // Time conflict detected
		}

		// Schedule activity
		if (activity_type == "class") {
			Array action;
			action.push_back("action_attend_class");
			action.push_back(activity["class_id"]);
			action.push_back(timeslot);
			subtasks.push_back(action);
		} else if (activity_type == "study") {
			Array action;
			action.push_back("action_study");
			action.push_back(activity["subject"]);
			action.push_back(timeslot);
			subtasks.push_back(action);
		} else if (activity_type == "social") {
			Array action;
			action.push_back("action_talk_to_character");
			action.push_back(activity["character_id"]);
			action.push_back(timeslot);
			subtasks.push_back(action);
		} else if (activity_type == "event") {
			Array action;
			action.push_back("action_participate_event");
			action.push_back(activity["event_id"]);
			action.push_back(timeslot);
			subtasks.push_back(action);
		}
	}

	return subtasks;
}

static Variant task_method_trigger_event(Dictionary p_state, String p_event_id) {
	Dictionary events = p_state["events"];
	Dictionary completed_events;
	if (p_state.has("completed_events")) {
		completed_events = p_state["completed_events"];
	}

	if (!events.has(p_event_id)) {
		return false;
	}

	if (completed_events.has(p_event_id) && completed_events[p_event_id].operator bool()) {
		return Array(); // Already completed
	}

	Dictionary event_info = events[p_event_id];

	// Check prerequisites
	if (event_info.has("prerequisites")) {
		Array prerequisites = event_info["prerequisites"];
		Dictionary story_flags = p_state["story_flags"];
		for (int i = 0; i < prerequisites.size(); i++) {
			String prereq_flag = prerequisites[i];
			if (!story_flags.has(prereq_flag) || !story_flags[prereq_flag].operator bool()) {
				// Need to complete prerequisite
				Array subtasks;
				Array task;
				task.push_back("trigger_event");
				// Find prerequisite event (simplified)
				subtasks.push_back(task);
				return subtasks;
			}
		}
	}

	// Find available timeslot
	Dictionary schedule = p_state["schedule"];
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}
	int64_t available_timeslot = -1;
	for (int64_t slot = 0; slot < 24; slot++) {
		if (!player_schedule.has(slot) || !player_schedule[slot].operator bool()) {
			available_timeslot = slot;
			break;
		}
	}

	if (available_timeslot == -1) {
		return false; // No available timeslot
	}

	Array subtasks;
	Array action;
	action.push_back("action_participate_event");
	action.push_back(p_event_id);
	action.push_back(available_timeslot);
	subtasks.push_back(action);
	return subtasks;
}

// Goal methods
static Variant goal_method_relationship_level(Dictionary p_state, String p_character_id, Variant p_desired_level) {
	Dictionary relationships = p_state["relationships"];
	int current_level = relationships.has(p_character_id) ? relationships[p_character_id].operator int() : 0;
	int target_level = p_desired_level.operator int();

	if (current_level >= target_level) {
		return Array(); // Goal already achieved
	}

	// Need to build relationship
	Array subtasks;
	Array task;
	task.push_back("build_relationship");
	task.push_back(p_character_id);
	task.push_back(target_level);
	subtasks.push_back(task);
	return subtasks;
}

static Variant goal_method_relationship_level_alternative(Dictionary p_state, String p_character_id, Variant p_desired_level) {
	// Alternative method that might fail
	Dictionary characters = p_state["characters"];
	if (!characters.has(p_character_id)) {
		return false; // Character doesn't exist, this method fails
	}

	// Try to build relationship
	Array subtasks;
	Array task;
	task.push_back("build_relationship");
	task.push_back(p_character_id);
	task.push_back(p_desired_level.operator int());
	subtasks.push_back(task);
	return subtasks;
}

static Variant goal_method_event_completed(Dictionary p_state, String p_event_id, Variant p_desired_value) {
	Dictionary completed_events;
	if (p_state.has("completed_events")) {
		completed_events = p_state["completed_events"];
	}
	bool is_completed = completed_events.has(p_event_id) && completed_events[p_event_id].operator bool();

	if (is_completed == p_desired_value.operator bool()) {
		return Array(); // Goal already achieved
	}

	// Need to trigger event
	Array subtasks;
	Array task;
	task.push_back("trigger_event");
	task.push_back(p_event_id);
	subtasks.push_back(task);
	return subtasks;
}

static Variant goal_method_time_slot_used(Dictionary p_state, int64_t p_timeslot, Variant p_desired_value) {
	Dictionary schedule = p_state["schedule"];
	Dictionary player_schedule;
	if (schedule.has("player")) {
		player_schedule = schedule["player"];
	}
	bool is_used = player_schedule.has(p_timeslot) && player_schedule[p_timeslot].operator bool();

	if (is_used == p_desired_value.operator bool()) {
		return Array(); // Goal already achieved
	}

	// Cannot directly set timeslot usage - would need an activity
	return false;
}

// Multigoal methods
static Variant multigoal_method_story_route(Dictionary p_state, Dictionary p_multigoal) {
	// Multigoal format: {"relationships": {"char1": 5, "char2": 3}, "events": {"event1": true, "event2": true}}
	Dictionary goal_relationships = PlannerMultigoal::get_goal_conditions_for_variable(p_multigoal, "relationships");
	Dictionary goal_events = PlannerMultigoal::get_goal_conditions_for_variable(p_multigoal, "events");

	Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, p_multigoal);
	if (goals_not_achieved.is_empty()) {
		return Array(); // All goals achieved
	}

	// Need to complete story route
	Array subtasks;
	Array task;
	task.push_back("complete_story_route");
	task.push_back("route1"); // Simplified
	subtasks.push_back(task);
	return subtasks;
}

// Setup academy visual novel domain
static Ref<PlannerDomain> setup_academy_visual_novel_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&action_talk_to_character));
	actions.push_back(callable_mp_static(&action_attend_class));
	actions.push_back(callable_mp_static(&action_study));
	actions.push_back(callable_mp_static(&action_participate_event));
	actions.push_back(callable_mp_static(&action_make_choice));
	actions.push_back(callable_mp_static(&action_spend_time_with));
	domain->add_actions(actions);

	// Add task methods
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&task_method_build_relationship));
	domain->add_task_methods("build_relationship", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_complete_story_route));
	domain->add_task_methods("complete_story_route", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_manage_time));
	domain->add_task_methods("manage_time", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_trigger_event));
	domain->add_task_methods("trigger_event", task_methods);

	// Add goal methods (with multiple methods for backtracking)
	TypedArray<Callable> goal_methods;
	goal_methods.push_back(callable_mp_static(&goal_method_relationship_level));
	goal_methods.push_back(callable_mp_static(&goal_method_relationship_level_alternative));
	domain->add_unigoal_methods("relationship_level", goal_methods);

	goal_methods.clear();
	goal_methods.push_back(callable_mp_static(&goal_method_event_completed));
	domain->add_unigoal_methods("event_completed", goal_methods);

	goal_methods.clear();
	goal_methods.push_back(callable_mp_static(&goal_method_time_slot_used));
	domain->add_unigoal_methods("time_slot_used", goal_methods);

	// Add multigoal methods
	TypedArray<Callable> multigoal_methods;
	multigoal_methods.push_back(callable_mp_static(&multigoal_method_story_route));
	domain->add_multigoal_methods(multigoal_methods);

	// Set up action dictionary
	Dictionary action_dict;
	action_dict["action_talk_to_character"] = callable_mp_static(&action_talk_to_character);
	action_dict["action_attend_class"] = callable_mp_static(&action_attend_class);
	action_dict["action_study"] = callable_mp_static(&action_study);
	action_dict["action_participate_event"] = callable_mp_static(&action_participate_event);
	action_dict["action_make_choice"] = callable_mp_static(&action_make_choice);
	action_dict["action_spend_time_with"] = callable_mp_static(&action_spend_time_with);
	domain->action_dictionary = action_dict;

	return domain;
}

} // namespace TestGameDomainsBacktracking
#endif // TOOLS_ENABLED
