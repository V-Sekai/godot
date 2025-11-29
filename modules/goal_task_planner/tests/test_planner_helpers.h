/**************************************************************************/
/*  test_planner_helpers.h                                                */
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

// Helper functions and domain definitions shared by planner tests.

#pragma once

#include "../planner_state.h"
#include "core/variant/callable.h"
#include "tests/test_macros.h"

namespace TestComprehensivePlanner {

// Wrapper class used to construct Callables for free functions in IsekaiAcademyDomain.
// Defined before IsekaiAcademyDomain namespace so it can be referenced from within it.
class IsekaiAcademyDomainCallable {
public:
	static Dictionary action_study_subject(Dictionary p_state, String p_student, String p_subject);
	static Dictionary action_attend_class(Dictionary p_state, String p_student, String p_class_name);
	static Dictionary action_talk_to_character(Dictionary p_state, String p_student, String p_character);
	static Dictionary action_increase_affection(Dictionary p_state, String p_student, String p_character, int p_amount);
	static Array task_complete_lesson(Dictionary p_state, String p_student, String p_subject);
	static Array task_build_relationship(Dictionary p_state, String p_student, String p_character);
	static Array unigoal_achieve_affection_level(Dictionary p_state, String p_relationship_key, int p_target_level);
	static Array unigoal_pass_exam(Dictionary p_state, String p_exam_key, bool p_target_value);
	static Array multigoal_complete_route(Dictionary p_state, Array p_multigoal);
};

// Helper functions for isekai academy visual novel domain
namespace IsekaiAcademyDomain {

// Actions
Dictionary action_study_subject(Dictionary state, String student, String subject) {
	Dictionary new_state = state.duplicate();
	Dictionary student_state;
	if (new_state.has(student)) {
		student_state = new_state[student];
	} else {
		student_state = Dictionary();
	}
	Dictionary studies;
	if (student_state.has("studies")) {
		studies = student_state["studies"];
	} else {
		studies = Dictionary();
	}
	studies[subject] = true;
	student_state["studies"] = studies;
	new_state[student] = student_state;
	return new_state;
}

Dictionary action_attend_class(Dictionary state, String student, String class_name) {
	Dictionary new_state = state.duplicate();
	Dictionary student_state;
	if (new_state.has(student)) {
		student_state = new_state[student];
	} else {
		student_state = Dictionary();
	}
	Array classes_attended;
	if (student_state.has("classes_attended")) {
		classes_attended = student_state["classes_attended"];
	} else {
		classes_attended = Array();
	}
	if (!classes_attended.has(class_name)) {
		classes_attended.push_back(class_name);
	}
	student_state["classes_attended"] = classes_attended;
	new_state[student] = student_state;
	return new_state;
}

Dictionary action_talk_to_character(Dictionary state, String student, String character) {
	Dictionary new_state = state.duplicate();
	Dictionary relationship_state;
	if (new_state.has("relationships")) {
		relationship_state = new_state["relationships"];
	} else {
		relationship_state = Dictionary();
	}
	String relationship_key = student + "_" + character;
	Dictionary relationship;
	if (relationship_state.has(relationship_key)) {
		relationship = relationship_state[relationship_key];
	} else {
		relationship = Dictionary();
	}
	relationship["last_interaction"] = true;
	relationship_state[relationship_key] = relationship;
	new_state["relationships"] = relationship_state;
	return new_state;
}

Dictionary action_increase_affection(Dictionary state, String student, String character, int amount) {
	Dictionary new_state = state.duplicate();
	
	// Update nested relationship structure (for compatibility)
	Dictionary relationship_state;
	if (new_state.has("relationships")) {
		relationship_state = new_state["relationships"];
	} else {
		relationship_state = Dictionary();
	}
	String relationship_key = student + "_" + character;
	Dictionary relationship;
	if (relationship_state.has(relationship_key)) {
		relationship = relationship_state[relationship_key];
	} else {
		relationship = Dictionary();
		relationship["affection"] = 0;
	}
	int current_affection = relationship.get("affection", 0);
	relationship["affection"] = current_affection + amount;
	relationship_state[relationship_key] = relationship;
	new_state["relationships"] = relationship_state;
	
	// Also maintain flat affection structure for unigoal checking: state["affection"][key] = value
	Dictionary affection_dict;
	if (new_state.has("affection")) {
		affection_dict = new_state["affection"];
	} else {
		affection_dict = Dictionary();
	}
	int new_affection = current_affection + amount;
	affection_dict[relationship_key] = new_affection;
	new_state["affection"] = affection_dict;
	
	return new_state;
}

// Task methods
Array task_complete_lesson(Dictionary state, String student, String subject) {
	Array subtasks;
	// Return actions as arrays: ["action_name", arg1, arg2, ...]
	Array action1;
	action1.push_back("action_study_subject");
	action1.push_back(student);
	action1.push_back(subject);
	subtasks.push_back(action1);
	
	Array action2;
	action2.push_back("action_attend_class");
	action2.push_back(student);
	action2.push_back(subject + "_class");
	subtasks.push_back(action2);
	return subtasks;
}

Array task_build_relationship(Dictionary state, String student, String character) {
	Array subtasks;
	// Return actions as arrays: ["action_name", arg1, arg2, ...]
	Array action1;
	action1.push_back("action_talk_to_character");
	action1.push_back(student);
	action1.push_back(character);
	subtasks.push_back(action1);
	
	Array action2;
	action2.push_back("action_increase_affection");
	action2.push_back(student);
	action2.push_back(character);
	action2.push_back(10);
	subtasks.push_back(action2);
	return subtasks;
}

// Unigoal methods
// Unigoal format: [predicate, subject, value]
// For affection: predicate="affection", subject="student_character" (e.g., "protagonist_class_president"), value=target_affection_level (e.g., 50)
Array unigoal_achieve_affection_level(Dictionary state, String relationship_key, int target_level) {
	Array subtasks;
	
	// Check current affection level: state["affection"][relationship_key]
	Dictionary affection_dict = state.get("affection", Dictionary());
	int current_affection = affection_dict.get(relationship_key, 0);
	
	if (current_affection < target_level) {
		// Extract student and character from relationship_key (format: "student_character")
		PackedStringArray parts = relationship_key.split("_");
		if (parts.size() >= 2) {
			String student = parts[0];
			// Rejoin the rest as character name (in case character name has underscores)
			String character = "";
			for (int i = 1; i < parts.size(); i++) {
				if (i > 1) {
					character += "_";
				}
				character += parts[i];
			}
			
			// Return one task - planner will re-check unigoal after execution
			Array task;
			task.push_back("build_relationship");
			task.push_back(student);
			task.push_back(character);
			subtasks.push_back(task);
		}
	}
	// If current_affection >= target_level, return empty array (unigoal achieved)
	return subtasks;
}

// Unigoal format: [predicate, subject, value]
// For pass_exam: predicate="exam_passed", subject="student_subject" (e.g., "protagonist_magic_class"), value=true
Array unigoal_pass_exam(Dictionary state, String exam_key, bool target_value) {
	Array subtasks;
	
	// Check current exam status: state["exam_passed"][exam_key]
	Dictionary exam_dict = state.get("exam_passed", Dictionary());
	bool current_status = exam_dict.get(exam_key, false);
	
	if (current_status != target_value) {
		// Extract student and subject from exam_key (format: "student_subject")
		PackedStringArray parts = exam_key.split("_");
		if (parts.size() >= 2) {
			String student = parts[0];
			// Rejoin the rest as subject name (in case subject name has underscores)
			String subject = "";
			for (int i = 1; i < parts.size(); i++) {
				if (i > 1) {
					subject += "_";
				}
				subject += parts[i];
			}
			
			// Return one task - planner will re-check unigoal after execution
			Array task;
			task.push_back("complete_lesson");
			task.push_back(student);
			task.push_back(subject);
			subtasks.push_back(task);
		}
	}
	// If current_status == target_value, return empty array (unigoal achieved)
	return subtasks;
}

// Multigoal method - multigoal is now an Array of unigoal arrays
// This method just returns the multigoal as-is (it's already an Array of unigoals)
Array multigoal_complete_route(Dictionary state, Array multigoal) {
	// Multigoal is already an Array of unigoal arrays, so just return it
	// This allows the planner to process each unigoal in order
	return multigoal;
}

} // namespace IsekaiAcademyDomain

// Implementations of IsekaiAcademyDomainCallable static methods
inline Dictionary IsekaiAcademyDomainCallable::action_study_subject(Dictionary p_state, String p_student, String p_subject) {
	return IsekaiAcademyDomain::action_study_subject(p_state, p_student, p_subject);
}

inline Dictionary IsekaiAcademyDomainCallable::action_attend_class(Dictionary p_state, String p_student, String p_class_name) {
	return IsekaiAcademyDomain::action_attend_class(p_state, p_student, p_class_name);
}

inline Dictionary IsekaiAcademyDomainCallable::action_talk_to_character(Dictionary p_state, String p_student, String p_character) {
	return IsekaiAcademyDomain::action_talk_to_character(p_state, p_student, p_character);
}

inline Dictionary IsekaiAcademyDomainCallable::action_increase_affection(Dictionary p_state, String p_student, String p_character, int p_amount) {
	return IsekaiAcademyDomain::action_increase_affection(p_state, p_student, p_character, p_amount);
}

inline Array IsekaiAcademyDomainCallable::task_complete_lesson(Dictionary p_state, String p_student, String p_subject) {
	return IsekaiAcademyDomain::task_complete_lesson(p_state, p_student, p_subject);
}

inline Array IsekaiAcademyDomainCallable::task_build_relationship(Dictionary p_state, String p_student, String p_character) {
	return IsekaiAcademyDomain::task_build_relationship(p_state, p_student, p_character);
}

inline Array IsekaiAcademyDomainCallable::unigoal_achieve_affection_level(Dictionary p_state, String p_relationship_key, int p_target_level) {
	return IsekaiAcademyDomain::unigoal_achieve_affection_level(p_state, p_relationship_key, p_target_level);
}

inline Array IsekaiAcademyDomainCallable::unigoal_pass_exam(Dictionary p_state, String p_exam_key, bool p_target_value) {
	return IsekaiAcademyDomain::unigoal_pass_exam(p_state, p_exam_key, p_target_value);
}

inline Array IsekaiAcademyDomainCallable::multigoal_complete_route(Dictionary p_state, Array p_multigoal) {
	return IsekaiAcademyDomain::multigoal_complete_route(p_state, p_multigoal);
}

// Helper function to validate a plan result
// Returns true if result is a valid plan (Array, not false)
// Optionally checks that plan is not empty when expect_non_empty is true
static bool is_valid_plan_result(Variant result, bool expect_non_empty = false) {
	if (result.get_type() == Variant::BOOL) {
		// false means planning failed - this is NOT a valid plan
		return false;
	}
	if (result.get_type() != Variant::ARRAY) {
		// Should be an array
		return false;
	}
	Array plan = result;
	if (expect_non_empty && plan.is_empty()) {
		// Expected non-empty plan but got empty - this is NOT a valid plan
		return false;
	}
	return true;
}

// Helper function to check plan contains expected action
// Plan is Array of action arrays like [["action_name", arg1, arg2], ...]
static bool plan_contains_action(Array plan, String action_name) {
	for (int i = 0; i < plan.size(); i++) {
		Variant item = plan[i];
		if (item.get_type() == Variant::ARRAY) {
			Array action = item;
			if (!action.is_empty()) {
				Variant first = action[0];
				// Handle both string and unwrapped dictionary cases
				if (first.get_type() == Variant::STRING && first == action_name) {
					return true;
				}
				// Handle dictionary-wrapped actions
				if (first.get_type() == Variant::DICTIONARY) {
					Dictionary dict = first;
					if (dict.has("item")) {
						Variant unwrapped = dict["item"];
						if (unwrapped.get_type() == Variant::ARRAY) {
							Array unwrapped_arr = unwrapped;
							if (!unwrapped_arr.is_empty() && unwrapped_arr[0] == action_name) {
								return true;
							}
						}
					}
				}
			}
		}
	}
	return false;
}

// Helper to validate plan structure matches expected fixture
// expected_min_actions: minimum number of actions expected in plan
// expected_actions: array of action names that should be present
static bool validate_plan_against_fixture(Array plan, int expected_min_actions, Array expected_actions = Array()) {
	if (plan.size() < expected_min_actions) {
		return false;
	}
	
	// Check that expected actions are present
	for (int i = 0; i < expected_actions.size(); i++) {
		String action_name = expected_actions[i];
		if (!plan_contains_action(plan, action_name)) {
			return false;
		}
	}
	
	return true;
}

} // namespace TestComprehensivePlanner
