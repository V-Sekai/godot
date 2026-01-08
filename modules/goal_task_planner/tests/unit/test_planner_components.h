#pragma once

#include "modules/goal_task_planner/src/domain.h"
#include "modules/goal_task_planner/src/graph_operations.h"
#include "modules/goal_task_planner/src/multigoal.h"
#include "modules/goal_task_planner/src/planner_result.h"
#include "tests/test_macros.h"

namespace TestPlannerComponents {

TEST_CASE("[Modules][Planner][Result] PlannerResult Basics") {
    Ref<PlannerResult> result = memnew(PlannerResult);

    SUBCASE("Initialization") {
        CHECK_FALSE(result->get_success());
        CHECK(result->get_final_state().is_empty());
        CHECK(result->get_solution_graph().is_empty());
    }

    SUBCASE("Setters and Getters") {
        result->set_success(true);
        CHECK(result->get_success());

        Dictionary state;
        state["foo"] = "bar";
        result->set_final_state(state);
        CHECK(String(result->get_final_state()["foo"]) == "bar");
    }
}

TEST_CASE("[Modules][Planner][Multigoal] PlannerMultigoal Basics") {
    SUBCASE("Validation") {
        // Simple goal: [predicate, subject, value]
        Array simple_goal;
        simple_goal.push_back("loc");
        simple_goal.push_back("robot");
        simple_goal.push_back("hallway");
        CHECK_FALSE(PlannerMultigoal::is_multigoal_array(simple_goal));

        // Multigoal: [simple_goal, simple_goal]
        Array multigoal;
        multigoal.push_back(simple_goal);
        CHECK(PlannerMultigoal::is_multigoal_array(multigoal));
    }

    SUBCASE("Tags") {
        Array multigoal;
        Array simple_goal;
        simple_goal.push_back("loc");
        multigoal.push_back(simple_goal);

        // Set tag
        // method signature: set_goal_tag(Variant multigoal, String tag) -> Variant
        Variant tagged = PlannerMultigoal::set_goal_tag(multigoal, "my_goal");
        CHECK(tagged.get_type() == Variant::DICTIONARY);
        Dictionary tagged_dict = tagged;
        CHECK(String(tagged_dict["goal_tag"]) == "my_goal");
        CHECK(tagged_dict.has("multigoal"));

        // Get tag
        String retrieved_tag = PlannerMultigoal::get_goal_tag(tagged);
        CHECK(retrieved_tag == "my_goal");

        // Get tag from untagged array
        CHECK(PlannerMultigoal::get_goal_tag(multigoal) == "");
    }
}

TEST_CASE("[Modules][Planner][GraphOps] Node Types") {
    Dictionary action_dict;
    action_dict["move"] = true; // Dummy value, checking for existence
    Dictionary task_dict;
    task_dict["deliver"] = true;
    Dictionary unigoal_dict; // Usually empty/internal

    SUBCASE("Detect Command") {
        // Actions/Commands are typically strings in node info or arrays starting with string
        Variant node_info = "move";
        PlannerNodeType type = PlannerGraphOperations::get_node_type(node_info, action_dict, task_dict, unigoal_dict);
        CHECK(type == PlannerNodeType::TYPE_ACTION);

        Array node_info_arr;
        node_info_arr.push_back("move");
        node_info_arr.push_back("arg1");
        type = PlannerGraphOperations::get_node_type(node_info_arr, action_dict, task_dict, unigoal_dict);
        CHECK(type == PlannerNodeType::TYPE_ACTION);
    }

    SUBCASE("Detect Task") {
        Variant node_info = "deliver";
        PlannerNodeType type = PlannerGraphOperations::get_node_type(node_info, action_dict, task_dict, unigoal_dict);
        CHECK(type == PlannerNodeType::TYPE_TASK);
    }

    SUBCASE("Detect Multigoal") {
        // Multigoal is array of arrays
        Array goal;
        goal.push_back("loc");
        Array multigoal;
        multigoal.push_back(goal);
        
        PlannerNodeType type = PlannerGraphOperations::get_node_type(multigoal, action_dict, task_dict, unigoal_dict);
        CHECK(type == PlannerNodeType::TYPE_MULTIGOAL);
    }
}

} // namespace TestPlannerComponents
