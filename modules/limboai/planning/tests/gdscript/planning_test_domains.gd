# planning_test_domains.gd
# GDScript equivalents of the C++ planning test domains for use with godot --script.
# Commands: (state: Dictionary, ...args) -> Dictionary (new state or empty on failure).
# Unigoal methods: (state, subject, value) -> Array of planner elements (command arrays).
# Task methods: (state, ...task_args) -> Array of command arrays.

class_name PlanningTestDomains
extends RefCounted

# ---- Minimal goal / two-step (value predicate) ----

static func action_increment(state: Dictionary, amount: int) -> Dictionary:
	var new_state := state.duplicate(true)
	var current := 0
	if new_state.has("value"):
		var vd = new_state["value"]
		if vd is Dictionary and vd.has("value"):
			current = vd["value"]
	var value_dict := new_state.get("value", {}) as Dictionary
	value_dict = value_dict.duplicate(true)
	value_dict["value"] = current + amount
	new_state["value"] = value_dict
	return new_state

static func unigoal_value_method(_state: Dictionary, _subject: String, _value: Variant) -> Array:
	return [["action_increment", 1]]

static func unigoal_value_method_two_steps(_state: Dictionary, _subject: String, value: Variant) -> Array:
	if int(value) != 2:
		return []
	return [["action_increment", 1], ["action_increment", 1]]

func create_minimal_goal_domain() -> PlannerDomain:
	var domain := PlannerDomain.new()
	domain.add_command("action_increment", Callable(self, "_cmd_increment"))
	var methods: Array = [Callable(self, "_unigoal_value")]
	domain.add_unigoal_methods("value", methods)
	return domain

func create_two_step_goal_domain() -> PlannerDomain:
	var domain := PlannerDomain.new()
	domain.add_command("action_increment", Callable(self, "_cmd_increment"))
	var methods: Array = [Callable(self, "_unigoal_value"), Callable(self, "_unigoal_value_two_steps")]
	domain.add_unigoal_methods("value", methods)
	return domain

func _cmd_increment(state: Dictionary, amount: int) -> Dictionary:
	return action_increment(state, amount)

func _unigoal_value(state: Dictionary, subject: String, value: Variant) -> Array:
	return unigoal_value_method(state, subject, value)

func _unigoal_value_two_steps(state: Dictionary, subject: String, value: Variant) -> Array:
	return unigoal_value_method_two_steps(state, subject, value)

# ---- HTN and backtracking (increment task) ----

static func task_increment_succeed(state: Dictionary) -> Array:
	return [["action_increment", 1]]

static func task_increment_fail(_state: Dictionary) -> Variant:
	return null  # fail

func create_minimal_htn_domain() -> PlannerDomain:
	var domain := PlannerDomain.new()
	domain.add_command("action_increment", Callable(self, "_cmd_increment"))
	var methods: Array = [Callable(self, "_task_increment_succeed")]
	domain.add_task_methods("increment", methods)
	return domain

func create_minimal_backtracking_domain() -> PlannerDomain:
	var domain := PlannerDomain.new()
	domain.add_command("action_increment", Callable(self, "_cmd_increment"))
	var methods: Array = [Callable(self, "_task_increment_fail"), Callable(self, "_task_increment_succeed")]
	domain.add_task_methods("increment", methods)
	return domain

func _task_increment_succeed(state: Dictionary) -> Array:
	return task_increment_succeed(state)

func _task_increment_fail(state: Dictionary) -> Variant:
	return task_increment_fail(state)

# ---- Academy one-block domain (agent-based interaction) ----

const BLOCK := "block"
const AGENT_LOCATIONS := "agent_locations"
const KEY_AT := "key_at"
const ARCHIVE_LOCKED := "archive_locked"
const ARCHIVE_ACCESSIBLE := "archive_accessible"
const INVENTORY := "inventory"
const OUTFIT := "outfit"

static func _get_block(state: Dictionary) -> Dictionary:
	if not state.has(BLOCK):
		state[BLOCK] = {
			AGENT_LOCATIONS: {},
			KEY_AT: "lounge",
			ARCHIVE_LOCKED: true,
			ARCHIVE_ACCESSIBLE: false,
			INVENTORY: {},
			OUTFIT: {}
		}
	return state[BLOCK]

static func _agent_location(block: Dictionary, agent_id: String) -> String:
	var locs = block.get(AGENT_LOCATIONS, {})
	return locs.get(agent_id, "")

static func _agent_has_object(block: Dictionary, agent_id: String, object_id: String) -> bool:
	var inv = block.get(INVENTORY, {})
	if not inv.has(agent_id):
		return false
	var arr: Array = inv[agent_id]
	return object_id in arr

static func action_move(state: Dictionary, agent_id: String, location_id: String) -> Dictionary:
	var new_state := state.duplicate(true)
	var block := _get_block(new_state).duplicate(true)
	var locs: Dictionary = block[AGENT_LOCATIONS].duplicate(true)
	locs[agent_id] = location_id
	block[AGENT_LOCATIONS] = locs
	new_state[BLOCK] = block
	return new_state

static func action_take(state: Dictionary, agent_id: String, object_id: String) -> Dictionary:
	if object_id != "key":
		return {}
	var new_state := state.duplicate(true)
	var block := _get_block(new_state).duplicate(true)
	var key_at: String = block.get(KEY_AT, "lounge")
	var loc := _agent_location(block, agent_id)
	if key_at != "lounge" or loc != "lounge":
		return {}
	block[KEY_AT] = agent_id
	var inv: Dictionary = block[INVENTORY].duplicate(true)
	var agent_inv: Array = inv.get(agent_id, []).duplicate()
	agent_inv.append("key")
	inv[agent_id] = agent_inv
	block[INVENTORY] = inv
	new_state[BLOCK] = block
	return new_state

static func action_use_object(state: Dictionary, agent_id: String, object_id: String) -> Dictionary:
	var new_state := state.duplicate(true)
	var block := _get_block(new_state).duplicate(true)
	var loc := _agent_location(block, agent_id)
	if object_id == "key":
		if not _agent_has_object(block, agent_id, "key") or loc != "archive_door":
			return {}
		block[ARCHIVE_LOCKED] = false
		block[ARCHIVE_ACCESSIBLE] = true
		new_state[BLOCK] = block
		return new_state
	if object_id == "terminal":
		if loc != "terminal":
			return {}
		if state.has("entity_capabilities") and state.entity_capabilities.has(agent_id):
			var ec: Dictionary = state.entity_capabilities[agent_id]
			if ec.get("role", "") == "hack" or ec.get("hack", false):
				block[ARCHIVE_ACCESSIBLE] = true
				new_state[BLOCK] = block
				return new_state
		return {}
	return {}

static func action_interact_with(state: Dictionary, agent_id: String, target_id: String) -> Dictionary:
	if target_id != "librarian":
		return {}
	var new_state := state.duplicate(true)
	var block := _get_block(new_state).duplicate(true)
	var key_at: String = block.get(KEY_AT, "lounge")
	var loc := _agent_location(block, agent_id)
	if key_at != "librarian" or loc != "librarian_room":
		return {}
	block[KEY_AT] = agent_id
	var inv: Dictionary = block[INVENTORY].duplicate(true)
	var agent_inv: Array = inv.get(agent_id, []).duplicate()
	agent_inv.append("key")
	inv[agent_id] = agent_inv
	block[INVENTORY] = inv
	new_state[BLOCK] = block
	return new_state

static func action_equip_garment(state: Dictionary, agent_id: String, garment_id: String) -> Dictionary:
	var new_state := state.duplicate(true)
	var block := _get_block(new_state).duplicate(true)
	var outfit: Dictionary = block[OUTFIT].duplicate(true)
	outfit[agent_id] = garment_id
	block[OUTFIT] = outfit
	new_state[BLOCK] = block
	return new_state

static func task_prepare_student(state: Dictionary, agent_id: String) -> Array:
	var garment := "apprentice_cloak"
	if state.has("entity_capabilities") and state.entity_capabilities.has(agent_id):
		var ec: Dictionary = state.entity_capabilities[agent_id]
		if ec.get("role", "") == "bio_steward":
			garment = "apron_of_abundance"
	return [["action_equip_garment", agent_id, garment]]

static func method_get_archive_via_librarian(_state: Dictionary, agent_id: String) -> Array:
	return [
		["action_move", agent_id, "librarian_room"],
		["action_interact_with", agent_id, "librarian"],
		["action_move", agent_id, "archive_door"],
		["action_use_object", agent_id, "key"]
	]

static func method_get_archive_via_lounge(_state: Dictionary, agent_id: String) -> Array:
	return [
		["action_move", agent_id, "lounge"],
		["action_take", agent_id, "key"],
		["action_move", agent_id, "archive_door"],
		["action_use_object", agent_id, "key"]
	]

static func method_get_archive_via_hack(_state: Dictionary, agent_id: String) -> Array:
	return [
		["action_move", agent_id, "terminal"],
		["action_use_object", agent_id, "terminal"]
	]

func create_academy_one_block_domain() -> PlannerDomain:
	var domain := PlannerDomain.new()
	domain.add_command("action_move", Callable(self, "_academy_action_move"))
	domain.add_command("action_take", Callable(self, "_academy_action_take"))
	domain.add_command("action_use_object", Callable(self, "_academy_action_use_object"))
	domain.add_command("action_interact_with", Callable(self, "_academy_action_interact_with"))
	domain.add_command("action_equip_garment", Callable(self, "_academy_action_equip_garment"))
	var prepare_methods: Array = [Callable(self, "_academy_task_prepare_student")]
	domain.add_task_methods("prepare_student", prepare_methods)
	# Lounge method first (key in lounge) so it succeeds with default state
	var archive_methods: Array = [
		Callable(self, "_academy_method_lounge"),
		Callable(self, "_academy_method_librarian"),
		Callable(self, "_academy_method_hack")
	]
	domain.add_task_methods("get_archive_access", archive_methods)
	return domain

# Same as create_academy_one_block_domain but get_archive_access only has the lounge method.
# Use in tests where the initial state has key in lounge; avoids method selection preferring
# the shorter (hack) method which then fails for role bio_steward and can trigger backtracking edge cases.
func create_academy_one_block_domain_get_archive_lounge_only() -> PlannerDomain:
	var domain := PlannerDomain.new()
	domain.add_command("action_move", Callable(self, "_academy_action_move"))
	domain.add_command("action_take", Callable(self, "_academy_action_take"))
	domain.add_command("action_use_object", Callable(self, "_academy_action_use_object"))
	domain.add_command("action_interact_with", Callable(self, "_academy_action_interact_with"))
	domain.add_command("action_equip_garment", Callable(self, "_academy_action_equip_garment"))
	domain.add_task_methods("prepare_student", [Callable(self, "_academy_task_prepare_student")])
	domain.add_task_methods("get_archive_access", [Callable(self, "_academy_method_lounge")])
	return domain

func _academy_action_move(s: Dictionary, a: String, l: String) -> Dictionary:
	return action_move(s, a, l)

func _academy_action_take(s: Dictionary, a: String, o: String) -> Dictionary:
	return action_take(s, a, o)

func _academy_action_use_object(s: Dictionary, a: String, o: String) -> Dictionary:
	return action_use_object(s, a, o)

func _academy_action_interact_with(s: Dictionary, a: String, t: String) -> Dictionary:
	return action_interact_with(s, a, t)

func _academy_action_equip_garment(s: Dictionary, a: String, g: String) -> Dictionary:
	return action_equip_garment(s, a, g)

func _academy_task_prepare_student(s: Dictionary, a: String) -> Array:
	return task_prepare_student(s, a)

func _academy_method_librarian(s: Dictionary, a: String) -> Array:
	return method_get_archive_via_librarian(s, a)

func _academy_method_lounge(s: Dictionary, a: String) -> Array:
	return method_get_archive_via_lounge(s, a)

func _academy_method_hack(s: Dictionary, a: String) -> Array:
	return method_get_archive_via_hack(s, a)

# ---- IPyHOP backtracking_test (migrated from ipyhop_tests/backtracking_test.py) ----
# State: ipyhop.flag (int). Blackboard must have "ipyhop": {"flag": -1} (PlannerState only merges dict vars).
# Actions: a_putv(flag_val), a_getv(flag_val). Tasks: put_it, need0, need1, need01, need10.

static func _ipyhop_flag(state: Dictionary) -> int:
	var d = state.get("ipyhop", {})
	if d is Dictionary:
		return d.get("flag", -999)
	return -999

static func ipyhop_a_putv(state: Dictionary, flag_val: int) -> Dictionary:
	var new_state := state.duplicate(true)
	var box: Dictionary = (new_state.get("ipyhop", {}) as Dictionary).duplicate(true)
	box["flag"] = flag_val
	new_state["ipyhop"] = box
	return new_state

static func ipyhop_a_getv(state: Dictionary, flag_val: int):
	if _ipyhop_flag(state) != flag_val:
		return false
	var new_state := state.duplicate(true)
	var box: Dictionary = (new_state.get("ipyhop", {}) as Dictionary).duplicate(true)
	box["flag"] = flag_val
	new_state["ipyhop"] = box
	return new_state

static func ipyhop_m_err(_state: Dictionary) -> Array:
	return [["a_putv", 0], ["a_getv", 1]]

static func ipyhop_m0(_state: Dictionary) -> Array:
	return [["a_putv", 0], ["a_getv", 0]]

static func ipyhop_m1(_state: Dictionary) -> Array:
	return [["a_putv", 1], ["a_getv", 1]]

static func ipyhop_m_need0(_state: Dictionary) -> Array:
	return [["a_getv", 0]]

static func ipyhop_m_need1(_state: Dictionary) -> Array:
	return [["a_getv", 1]]

func create_ipyhop_backtracking_domain() -> PlannerDomain:
	var domain := PlannerDomain.new()
	domain.add_command("a_putv", Callable(self, "_ipyhop_a_putv"))
	domain.add_command("a_getv", Callable(self, "_ipyhop_a_getv"))
	domain.add_task_methods("put_it", [
		Callable(self, "_ipyhop_m_err"),
		Callable(self, "_ipyhop_m0"),
		Callable(self, "_ipyhop_m1")
	])
	domain.add_task_methods("need0", [Callable(self, "_ipyhop_m_need0")])
	domain.add_task_methods("need1", [Callable(self, "_ipyhop_m_need1")])
	domain.add_task_methods("need01", [Callable(self, "_ipyhop_m_need0"), Callable(self, "_ipyhop_m_need1")])
	domain.add_task_methods("need10", [Callable(self, "_ipyhop_m_need1"), Callable(self, "_ipyhop_m_need0")])
	return domain

func _ipyhop_a_putv(s: Dictionary, v: int) -> Dictionary:
	return ipyhop_a_putv(s, v)

func _ipyhop_a_getv(s: Dictionary, v: int) -> Variant:
	return ipyhop_a_getv(s, v)

func _ipyhop_m_err(s: Dictionary) -> Array:
	return ipyhop_m_err(s)

func _ipyhop_m0(s: Dictionary) -> Array:
	return ipyhop_m0(s)

func _ipyhop_m1(s: Dictionary) -> Array:
	return ipyhop_m1(s)

func _ipyhop_m_need0(s: Dictionary) -> Array:
	return ipyhop_m_need0(s)

func _ipyhop_m_need1(s: Dictionary) -> Array:
	return ipyhop_m_need1(s)
