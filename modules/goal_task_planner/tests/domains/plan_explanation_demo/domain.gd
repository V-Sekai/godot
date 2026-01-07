# Simple domain for demonstrating plan explanation and debugging features
# This domain models a simple "go to store and buy item" scenario

class_name PlanExplanationDemoDomain

static func create_planner_domain() -> PlannerDomain:
	var domain = PlannerDomain.new()
	
	# Register actions
	var actions = []
	actions.append(Callable(PlanExplanationDemoDomain, "action_walk"))
	actions.append(Callable(PlanExplanationDemoDomain, "action_buy"))
	domain.add_actions(actions)
	
	# Register task methods
	var go_to_methods = []
	go_to_methods.append(Callable(PlanExplanationDemoDomain, "method_go_to_direct"))
	go_to_methods.append(Callable(PlanExplanationDemoDomain, "method_go_to_via_landmark"))
	domain.add_task_methods("go_to", go_to_methods)
	
	var buy_methods = []
	buy_methods.append(Callable(PlanExplanationDemoDomain, "method_buy_item"))
	domain.add_task_methods("buy", buy_methods)
	
	return domain

# Actions
static func action_walk(state: Dictionary, from: String, to: String) -> Variant:
	if state.get("location") != from:
		return false
	
	var new_state = state.duplicate()
	new_state["location"] = to
	return new_state

static func action_buy(state: Dictionary, item: String, store: String) -> Variant:
	if state.get("location") != store:
		return false
	
	if not state.get("money", 0) >= 10:
		return false
	
	var new_state = state.duplicate()
	new_state["money"] = state.get("money", 0) - 10
	if not new_state.has("inventory"):
		new_state["inventory"] = []
	var inventory = new_state["inventory"].duplicate()
	inventory.append(item)
	new_state["inventory"] = inventory
	return new_state

# Task Methods
static func method_go_to_direct(state: Dictionary, destination: String) -> Variant:
	var current = state.get("location", "home")
	if current == destination:
		return []
	
	# Direct path - faster but might not always work
	if destination == "store" and current == "home":
		return [["walk", "home", "store"]]
	
	return null

static func method_go_to_via_landmark(state: Dictionary, destination: String) -> Variant:
	var current = state.get("location", "home")
	if current == destination:
		return []
	
	# Via landmark - slower but more reliable
	if destination == "store" and current == "home":
		return [
			["walk", "home", "park"],
			["walk", "park", "store"]
		]
	
	return null

static func method_buy_item(state: Dictionary, item: String) -> Variant:
	var location = state.get("location", "home")
	if location != "store":
		return [["go_to", "store"], ["buy", item, "store"]]
	
	return [["buy", item, "store"]]

