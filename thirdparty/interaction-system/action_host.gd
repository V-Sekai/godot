extends Node3D

const interaction_manager_class := preload("./interaction_manager.gd")

const base_action_class := preload("./controller_actions/base_action.gd")

class ActionState extends RefCounted:
	var base_action: base_action_class

	func _init(base_action_: base_action_class):
		base_action = base_action_

var child_action_set: Dictionary[base_action_class, ActionState]

var active_buttons: Dictionary[String, base_action_class]

var interaction_manager: interaction_manager_class

func _tracker_added():
	pass

func _tracker_removed():
	pass

func _ready():
	for node in get_children():
		var ba := node as base_action_class
		if ba == null:
			continue
		child_action_set[ba] = ActionState.new(ba)
		ba.interaction_manager = interaction_manager
		ba.on_action_added()

func _exit_tree():
	var nodes_to_remove := child_action_set.duplicate()
	child_action_set.clear()
	for ba in nodes_to_remove:
		ba.on_action_removed()
	

func _notification(what: int):
	match what:
		NOTIFICATION_CHILD_ORDER_CHANGED:
			if is_node_ready():
				var nodes_to_remove := child_action_set.duplicate()
				for node in get_children():
					var ba := node as base_action_class
					if ba == null:
						continue
					nodes_to_remove.erase(ba)
					if not child_action_set.has(ba):
						child_action_set[ba] = ActionState.new(ba)
						ba.interaction_manager = interaction_manager
						ba.on_action_added()
				for ba in nodes_to_remove:
					child_action_set.erase(ba)
					ba.on_action_removed()

func fire_button_event(mb: InputEventMouseButton):
	# callers must assign some sort of button id as mb.resource_name
	var key: StringName = mb.resource_name
	if not mb.resource_name:
		key = StringName(str(mb.button_index))
	if active_buttons.has(key):
		var keep_active: bool = active_buttons[key].on_button_event(mb)
		if not keep_active and not mb.pressed:
			active_buttons.erase(key)
	else:
		for child_action in child_action_set:
			var make_active: bool = child_action.on_button_event(mb)
			if make_active:
				active_buttons[key] = child_action
				break

func fire_pose_changed(pose: XRPose):
	for child_action in child_action_set:
		child_action.on_pose_changed(pose)
