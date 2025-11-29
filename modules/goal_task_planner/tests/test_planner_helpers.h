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
	static Array unigoal_achieve_affection_level(Dictionary p_state, String p_student, String p_character, int p_level);
	static Array unigoal_pass_exam(Dictionary p_state, String p_student, String p_subject);
	static Array multigoal_complete_route(Dictionary p_state, Dictionary p_multigoal);
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
	return new_state;
}

// Task methods
Array task_complete_lesson(Dictionary state, String student, String subject) {
	Array subtasks;
	subtasks.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_study_subject).bind(student, subject));
	subtasks.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_attend_class).bind(student, subject + "_class"));
	return subtasks;
}

Array task_build_relationship(Dictionary state, String student, String character) {
	Array subtasks;
	subtasks.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character).bind(student, character));
	subtasks.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_increase_affection).bind(student, character, 10));
	return subtasks;
}

// Unigoal methods
Array unigoal_achieve_affection_level(Dictionary state, String student, String character, int level) {
	Array subtasks;
	Dictionary relationship_state = state.get("relationships", Dictionary());
	String relationship_key = student + "_" + character;
	if (relationship_state.has(relationship_key)) {
		Dictionary relationship = relationship_state[relationship_key];
		int current_affection = relationship.get("affection", 0);
		if (current_affection < level) {
			subtasks.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::task_build_relationship).bind(student, character));
		}
	} else {
		subtasks.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::task_build_relationship).bind(student, character));
	}
	return subtasks;
}

Array unigoal_pass_exam(Dictionary state, String student, String subject) {
	Array subtasks;
	Dictionary student_state = state.get(student, Dictionary());
	Dictionary studies = student_state.get("studies", Dictionary());
	Array classes_attended = student_state.get("classes_attended", Array());
	if (!studies.has(subject) || !classes_attended.has(subject + "_class")) {
		subtasks.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::task_complete_lesson).bind(student, subject));
	}
	return subtasks;
}

// Multigoal method
Array multigoal_complete_route(Dictionary state, Dictionary multigoal) {
	Array result;
	Array characters = multigoal.keys();
	for (int i = 0; i < characters.size(); i++) {
		String character = characters[i];
		Dictionary character_goal = multigoal[character];
		int affection_level = character_goal.get("affection_level", 0);
		String student = character_goal.get("student", "");
		if (affection_level > 0 && !student.is_empty()) {
			result.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::unigoal_achieve_affection_level).bind(student, character, affection_level));
		}
	}
	return result;
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

inline Array IsekaiAcademyDomainCallable::unigoal_achieve_affection_level(Dictionary p_state, String p_student, String p_character, int p_level) {
	return IsekaiAcademyDomain::unigoal_achieve_affection_level(p_state, p_student, p_character, p_level);
}

inline Array IsekaiAcademyDomainCallable::unigoal_pass_exam(Dictionary p_state, String p_student, String p_subject) {
	return IsekaiAcademyDomain::unigoal_pass_exam(p_state, p_student, p_subject);
}

inline Array IsekaiAcademyDomainCallable::multigoal_complete_route(Dictionary p_state, Dictionary p_multigoal) {
	return IsekaiAcademyDomain::multigoal_complete_route(p_state, p_multigoal);
}

} // namespace TestComprehensivePlanner
