extends "./base_action.gd"

@export_node_path("MeshInstance3D") var laser_mesh: NodePath = ^"Laser":
	set(n):
		laser_mesh = n
		laser_mesh_instance = get_node_or_null(n) as MeshInstance3D
@onready var laser_mesh_instance: MeshInstance3D = get_node_or_null(laser_mesh) as MeshInstance3D

var query := interaction_manager_class.LassoQuery.new()

var current_poi: interaction_manager_class.LassoPOI
var current_canvas_item: CanvasItem
var focused_control: Control
var current_canvas_plane: interaction_manager_class.canvas_plane_class
var current_pos_2d: Vector2
var current_target_pos_3d: Vector3
var pressed_buttons: Dictionary[StringName, bool]

func on_action_added() -> void:
	pass

func on_action_removed() -> void:
	pass


func handle_pointer_moved_2d(last_canvas_item: CanvasItem, new_canvas_item: CanvasItem, pos_2d: Vector2) -> void:
	if new_canvas_item != null:
		var viewport: Viewport = new_canvas_item.get_viewport()
		if viewport != null:
			var event := InputEventMouseMotion.new()
			event.global_position = pos_2d.floor()
			event.position = pos_2d.floor()
			#print(center)
			event.set_relative(Vector2(0,0))  # should this be scaled/warped?
			event.set_button_mask(0)
			event.set_pressure(1.0)

			#if viewport:
			#	# viewport.push_input(event, true)
			#	viewport.push_unhandled_input(event, true)
			if new_canvas_item != last_canvas_item:
				var last_control := last_canvas_item as Control
				if last_control != null:
					var mouse_ref: int = last_control.get_meta(&"mouse_ref") - 1
					last_control.set_meta(&"mouse_ref", mouse_ref)
					if mouse_ref == 0:
						last_control.mouse_exited.emit()
						last_control.notify_thread_safe(Control.NOTIFICATION_MOUSE_EXIT)
				var new_control := new_canvas_item as Control
				if new_control != null:
					if not new_control.has_meta(&"mouse_ref"):
						new_control.set_meta(&"mouse_ref", 0)
					var mouse_ref: int = new_control.get_meta(&"mouse_ref") + 1
					new_control.set_meta(&"mouse_ref", mouse_ref)
					if mouse_ref == 1:
						new_control.mouse_entered.emit()
						new_control.notify_thread_safe(Control.NOTIFICATION_MOUSE_ENTER)
			var control := new_canvas_item as Control
			if control != null:
				# print (str(control.get_global_transform_with_canvas().origin) + " | " + str(event.global_position) + " | " + str(event.position) + " | " + str(control.name))
				event.position = event.global_position - control.get_global_transform_with_canvas().origin
				control.call_gui_input(event)
	else:
		var last_control := last_canvas_item as Control
		if last_control != null:
			var mouse_ref: int = last_control.get_meta(&"mouse_ref") - 1
			last_control.set_meta(&"mouse_ref", mouse_ref)
			if mouse_ref == 0:
				last_control.mouse_exited.emit()
				last_control.notify_thread_safe(Control.NOTIFICATION_MOUSE_EXIT)

# We ask the caller to pass in the coordinate for each event
# And delegate our own MOUSE_ENTERED and MOUSE_EXITED signals for each control.

# For example, a user should be able to select a different slider with each hand.
func handle_mouse_button(last_canvas_item: CanvasItem, mb: InputEventMouseButton):
	var ev := InputEventMouseButton.new()
	ev.global_position = mb.global_position
	ev.position = mb.position
	#ev.set_relative(Vector2(0,0))  # should this be scaled/warped?
	ev.double_click = mb.double_click
	ev.factor = mb.factor
	ev.pressed = mb.pressed
	ev.canceled = mb.canceled
	ev.button_index = mb.button_index # MouseButton.MOUSE_BUTTON_LEFT
	ev.button_mask = mb.button_mask  # MouseButtonMask.MOUSE_BUTTON_MASK_LEFT #
	var new_button := last_canvas_item as BaseButton

	var new_control := last_canvas_item as Control

	if new_control != null:
		var viewport := new_control.get_viewport()
		if viewport:
			viewport.push_unhandled_input(ev, true)
			# viewport.push_input(ev, true)
	if new_control != null:
		if focused_control != new_control:
			if focused_control != null:
				var focus_ref: int = focused_control.get_meta(&"focus_ref") - 1
				focused_control.set_meta(&"focus_ref", focus_ref)
				if focus_ref == 0:
					focused_control.notify_thread_safe(Control.NOTIFICATION_FOCUS_EXIT)
					#focused_control.focus_exited.emit()
			# new_control.notify_thread_safe(Control.NOTIFICATION_FOCUS_ENTER)
			if not new_control.has_meta(&"focus_ref"):
				new_control.set_meta(&"focus_ref", 0)
			var focus_ref: int = new_control.get_meta(&"focus_ref") + 1
			new_control.set_meta(&"focus_ref", focus_ref)
			if focus_ref == 1:
				new_control.grab_focus()
				new_control.grab_click_focus()
				#new_control.focus_entered.emit()
				focused_control = new_control
		var viewport = last_canvas_item.get_viewport()

		#ev.set_global_position(center)
		#ev.set_relative(Vector2(0,0))  # should this be scaled/warped?
		#ev.double_click = mb.double_click
		#ev.factor = mb.factor
		#ev.canceled = mb.canceled
		#ev.button_index = mb.button_index
		#ev.set_button_mask(mb.button_mask)
		ev.position = mb.global_position - new_control.get_global_transform_with_canvas().origin
		new_control.call_gui_input(ev)

		# var new_button := last_canvas_item as BaseButton
		#if new_button != null:
		#	new_button.button_pressed = ev.pressed
		#	new_button.set_pressed_no_signal(ev.pressed)
		#	new_button.pressed.emit()
		#	#new_button.shortcut_input
		#	#if ev.pressed:
		#		#print(InputMap.action_get_events(&"ui_accept"))
		#		#viewport.push_input(InputMap.action_get_events(&"ui_accept")[0])
	elif focused_control != null:
		var focus_ref: int = focused_control.get_meta(&"focus_ref") - 1
		focused_control.set_meta(&"focus_ref", focus_ref)
		if focus_ref == 0:
			focused_control.notify_thread_safe(Control.NOTIFICATION_FOCUS_EXIT)
			#focused_control.focus_exited.emit()
		focused_control = null


func on_button_event(mb: InputEventMouseButton) -> bool:
	var xform := transform
	if mb.pressed:
		pressed_buttons[mb.resource_name] = true
	else:
		pressed_buttons.erase(mb.resource_name)
	handle_pointer_moved_2d(null, current_canvas_item, current_pos_2d)
	mb.global_position = current_pos_2d.round()
	mb.position = current_pos_2d.round()
	handle_mouse_button(current_canvas_item, mb)
	return mb.pressed

func on_pose_changed(pose: XRPose):
	super.on_pose_changed(pose)
	if pose.name != primary_pose_name:
		return

	query.source = transform
	if not pressed_buttons.is_empty():
		query.override_point_set[current_poi] = true
	else:
		query.override_point_set.clear()
	if not interaction_manager.query_pointer_3d(query):
		return

	current_poi = query.out_best_poi
	if current_poi == null:
		return
	current_canvas_plane = interaction_manager.get_canvas_plane_from_poi(query.out_best_poi)
	var old_canvas_item: CanvasItem = current_canvas_item
	current_canvas_item = interaction_manager.get_canvas_item_from_poi(query.out_best_poi)
	current_target_pos_3d = query.get_position_3d(current_poi)

	current_pos_2d = interaction_manager.get_position_on_canvas_plane(current_canvas_plane, current_target_pos_3d)
	handle_pointer_moved_2d(old_canvas_item, current_canvas_item, current_pos_2d)
	if laser_mesh_instance != null:
		# print("Target pos 3d = " + str(current_target_pos_3d) + " / Target 2d = " + str(current_pos_2d))
		laser_mesh_instance.set_instance_shader_parameter(&"target", laser_mesh_instance.global_transform.affine_inverse() * current_target_pos_3d)
