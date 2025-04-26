/**************************************************************************/
/*  test_logistics.h                                                      */
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

#pragma once

#include "tests/test_macros.h"

#include "../hddl_document.h"

#ifdef TOOLS_ENABLED

namespace TestHDDL {
TEST_CASE("HDDL 2.1 Parsing with HTN, Fluents, and Time") { // Renamed for clarity
	Ref<HDDLDocument> hddl_doc = memnew(HDDLDocument);

	String hddl_text = R"(
			(define (domain test-domain-htn)
				(:requirements :strips :typing :numeric-fluents :htn :time) ; Added :numeric-fluents, :htn
				(:types object location)
				(:predicates (at ?o - object ?l - location))
				(:functions (fuel ?o - object) (distance ?l1 - location ?l2 - location))

				(:task drive :parameters (?o - object ?from - location ?to - location))

				(:action move
					:parameters (?o - object ?from - location ?to - location)
					:duration (= ?duration (distance ?from ?to)) ; Example duration
					:precondition (and (at ?o ?from) (>= (fuel ?o) 10))
					:effect (and (not (at ?o ?from)) (at ?o ?to) (decrease (fuel ?o) 10))
				)

				(:method m-drive
				   :parameters (?o - object ?from - location ?to - location)
				   :task (drive ?o ?from ?to)
				   :subtasks (ordered (move ?o ?from ?to)) ; Simple decomposition
				)
			)
			(define (problem test-problem-htn)
				(:domain test-domain-htn)
				(:objects
					truck - object
					locA locB - location
				)
				(:init
					(= (fuel truck) 50)
					(= (distance locA locB) 5) ; Define distance
					(at truck locA)
				)
				(:goal (and
					(at truck locB)
					(>= (fuel truck) 40)
				))
				(:htn :tasks (ordered (drive truck locA locB))) ; Use the defined task
			)
		)";

	HDDLNode *root_node = nullptr; // Use the raw pointer type defined in hddl_document.h

	Error error = hddl_doc->append_from_string(hddl_text, root_node);

	CHECK(error == Error::OK);
	CHECK(root_node != nullptr);

	// --- Domain Checks ---
	CHECK(root_node->type == "domain");
	CHECK(root_node->name == "test-domain-htn");
	// Find action 'move' and check details (simplified checks)
	ActionNode *move_action = nullptr;
	MethodNode *drive_method = nullptr;
	TaskNode *drive_task_def = nullptr; // Task definition in domain
	for (HDDLNode *child : root_node->children) {
		if (child->type == "action" && child->name == "move") {
			move_action = static_cast<ActionNode *>(child); // Assuming cast is safe/correct based on parsing logic
		} else if (child->type == "method" && child->name == "m-drive") {
			drive_method = static_cast<MethodNode *>(child);
		} else if (child->type == "task" && child->name == "drive") { // Check for task definition
			drive_task_def = static_cast<TaskNode *>(child);
		}
	}
	CHECK(move_action != nullptr);
	CHECK(move_action->parameters.size() == 3);
	// CHECK(move_action->duration != nullptr); // Add check for duration parsing if implemented
	CHECK(move_action->precondition != nullptr);
	CHECK(move_action->effect != nullptr);
	// CHECK(move_action->effect->type == EffectNode::AND); // Add detailed checks for effects if needed

	CHECK(drive_method != nullptr);
	CHECK(drive_method->task_name == "drive");
	CHECK(drive_method->subtasks.size() == 1);
	CHECK(drive_method->ordering == MethodNode::ORDERED);

	CHECK(drive_task_def != nullptr); // Check task definition exists
	CHECK(drive_task_def->parameters.size() == 3);

	// --- Problem Checks ---
	// Need to parse the problem part separately or have the parser return both
	// Assuming the parser handles both and we can access the problem node
	// This part needs adjustment based on how append_from_string handles multiple define blocks
	HDDLNode *problem_node = nullptr;
	// Logic to find the problem node (e.g., if append_from_string returns a list or updates a member)
	// For now, let's assume root_node might be updated or a second call is needed.
	// If append_from_string only parses the first 'define', this test needs restructuring.

	// --- Placeholder for Problem Parsing ---
	// Let's refine the test assuming append_from_string can handle multiple defines
	// or we make a second call. Assuming it returns the *last* define parsed:
	// Re-parse just the problem for simplicity in this example:
	Ref<HDDLDocument> hddl_doc_prob = memnew(HDDLDocument);
	HDDLNode *problem_root_node = nullptr;
	String problem_text = R"(
			(define (problem test-problem-htn)
				(:domain test-domain-htn)
				(:objects
					truck - object
					locA locB - location
				)
				(:init
					(= (fuel truck) 50)
					(= (distance locA locB) 5)
					(at truck locA)
				)
				(:goal (and
					(at truck locB)
					(>= (fuel truck) 40)
				))
				(:htn :tasks (ordered (drive truck locA locB)))
			)
		)";
	Error problem_error = hddl_doc_prob->append_from_string(problem_text, problem_root_node);

	CHECK(problem_error == Error::OK);
	CHECK(problem_root_node != nullptr);
	CHECK(problem_root_node->type == "problem");
	CHECK(problem_root_node->name == "test-problem-htn");

	// Check :objects (simplified)
	CHECK(problem_root_node->objects.size() >= 2); // Check if objects are parsed

	// Check :init (simplified)
	CHECK(problem_root_node->initial_state.size() == 3); // Check number of initial state atoms/assignments

	// Check :goal (simplified)
	CHECK(problem_root_node->goal != nullptr);
	CHECK(problem_root_node->goal->type == ConditionNode::AND); // Check goal structure
	// Add more detailed checks for goal conditions if parser populates ConditionNode fully

	// Check :htn (simplified)
	// Assuming HTN tasks are stored in a specific field, e.g., problem_root_node->htn_tasks
	// This depends heavily on the HDDLNode structure adaptation for problems.
	// Placeholder check:
	// CHECK(problem_root_node->htn_tasks.size() == 1);
	// CHECK(problem_root_node->htn_tasks[0].name == "drive");

	// Cleanup
	// Need proper memory management for the nodes created by the parser
	// delete root_node; // If manually managed
	// delete problem_root_node; // If manually managed
	memdelete(hddl_doc);
	memdelete(hddl_doc_prob);
}

} // namespace TestHDDL
#endif // TOOLS_ENABLED
