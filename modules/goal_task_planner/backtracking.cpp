/**************************************************************************/
/*  backtracking.cpp                                                      */
/**************************************************************************/

#include "backtracking.h"
#include "graph_operations.h"

PlannerBacktracking::BacktrackResult PlannerBacktracking::backtrack(PlannerSolutionGraph p_graph, int p_parent_node_id, int p_current_node_id, Dictionary p_state, TypedArray<Variant> p_blacklisted_commands) {
	// Mark current node as failed
	p_graph.set_node_status(p_current_node_id, PlannerNodeStatus::STATUS_FAILED);
	
	// Remove descendants of the failed node
	PlannerGraphOperations::remove_descendants(p_graph, p_current_node_id);
	
	// Find the nearest ancestor that can be retried
	int new_parent_node_id = p_parent_node_id;
	
	// Traverse up the tree to find a node that can be retried
	while (new_parent_node_id >= 0) {
		Dictionary node = p_graph.get_node(new_parent_node_id);
		int node_type = node["type"];
		TypedArray<Callable> available_methods = node["available_methods"];
		
		// Check if this node has alternative methods
		bool can_retry = false;
		
		if (node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
		    node_type == static_cast<int>(PlannerNodeType::TYPE_GOAL) ||
		    node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
			// Check if there are available methods
			if (available_methods.size() > 0) {
				can_retry = true;
			}
		}
		
		if (can_retry) {
			// Found a node with available methods, retry it
			p_graph.set_node_status(new_parent_node_id, PlannerNodeStatus::STATUS_OPEN);
			
			BacktrackResult result;
			result.parent_node_id = PlannerGraphOperations::find_predecessor(p_graph, new_parent_node_id);
			result.current_node_id = new_parent_node_id;
			result.graph = p_graph;
			result.state = p_state;
			result.blacklisted_commands = p_blacklisted_commands;
			return result;
		} else {
			// No more methods, this node also fails, continue backtracking
			p_graph.set_node_status(new_parent_node_id, PlannerNodeStatus::STATUS_FAILED);
			new_parent_node_id = PlannerGraphOperations::find_predecessor(p_graph, new_parent_node_id);
		}
	}
	
	// Reached root, return failure
	BacktrackResult result;
	result.parent_node_id = -1;
	result.current_node_id = -1;
	result.graph = p_graph;
	result.state = p_state;
	result.blacklisted_commands = p_blacklisted_commands;
	return result;
}

