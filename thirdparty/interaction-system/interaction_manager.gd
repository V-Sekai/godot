extends Node

const lassodb_class := preload("./lassodb.gd")
const canvas_plane_class := preload("res://addons/canvas_plane/canvas_plane.gd")
const canvas_3d_anchor := preload("res://addons/canvas_plane/canvas_3d_anchor.gd")

var lasso_db: lassodb_class
var canvas_planes: Array[canvas_plane_class]

func _init():
	lasso_db = lassodb_class.new()

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	get_viewport().set_meta(&"interaction_manager", self)

func _enter_tree() -> void:
	get_viewport().set_meta(&"interaction_manager", self)


func register_node_3d(node: Node3D):
	var lasso_point := lassodb_class.PointOfInterest.new()
	lasso_point.register_point(lasso_db, node)


func register_canvas(cp: canvas_plane_class):
	if cp.has_meta(&"canvas_registered"):
		return
	cp.set_meta(&"canvas_registered", true)
	var root_control: Control = cp.control_root
	var form_element: Control = root_control.find_next_valid_focus()
	var already_added_items := {}
	var canvas_anchors := root_control.find_children("*", "Node3D", false, false)
	print(canvas_anchors)
	for anchor in canvas_anchors:
		if anchor is canvas_3d_anchor:
			var canvas_item := anchor.get_node_or_null(anchor.canvas_item_node_path)
			anchor.canvas_item_node_path = NodePath("../" + str(cp.get_path_to(canvas_item)))
			anchor.reparent(cp)
	while form_element != null:
		print("New form element: " + str(form_element))
		already_added_items[form_element] = true
		# Get 3D position say way canvas 3d anchor works.
		var form_3d_anchor := canvas_3d_anchor.new()
		form_3d_anchor.canvas_item_node_path = NodePath("../" + str(cp.get_path_to(form_element)))
		cp.add_child(form_3d_anchor)
		# Add to LassoDB
		register_node_3d(form_3d_anchor)
		# Jump to next element
		var form_element_new: Control = form_element.find_next_valid_focus()
		if already_added_items.has(form_element_new):
			# Godot restarts at the beginning once we have found all focusable nodes.
			break
		form_element = form_element_new
	# TODO: Figure out how to maintain the list of form elements as they change.
	# TODO: figure out how to improve data structure
	# TODO: it might be nice to know if you are over a canvas and show a different mouse cursor or pointer.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass


func register_action_host(action_host: Node3D):
	pass

func unregister_action_host(action_host: Node3D):
	pass

const LassoQuery = lassodb_class.LassoQuery
const LassoPOI = lassodb_class.PointOfInterest

func query_pointer_3d(query: LassoQuery) -> bool:
	return lasso_db.query(query)

func get_canvas_plane_from_poi(poi: LassoPOI) -> canvas_plane_class:
	if poi != null:
		var main_3d_anchor := poi.origin as canvas_3d_anchor
		if main_3d_anchor != null:
			return main_3d_anchor.spatial_canvas
	return null

func get_canvas_item_from_poi(poi: LassoPOI) -> CanvasItem:
	if poi != null:
		var main_3d_anchor := poi.origin as canvas_3d_anchor
		if main_3d_anchor != null:
			return main_3d_anchor.canvas_item
	return null

func get_position_on_canvas_plane(sc: canvas_plane_class, poi_position_3d: Vector3) -> Vector2:
	var center_3d := poi_position_3d * sc.global_transform / (sc.canvas_scale * 0.5)
	var center := Vector2(center_3d.x, center_3d.y) + Vector2(sc.canvas_size.x, 1.0 - sc.canvas_size.y) * Vector2(sc.canvas_anchor_x, sc.canvas_anchor_y)
	center = Vector2(0.25 + center.x, 1.25 - center.y)
	return center

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

			if viewport:
				viewport.push_input(event, true)
			if false: # ci != last_canvas_item:
				var last_control := last_canvas_item as Control
				if last_control != null:
					#last_control.mouse_exited.emit()
					last_control.notify_thread_safe(Control.NOTIFICATION_MOUSE_EXIT)
				var new_control := new_canvas_item as Control
				if new_control != null:
					#new_control.mouse_entered.emit()
					new_control.notify_thread_safe(Control.NOTIFICATION_MOUSE_ENTER)
	else:
		var last_control := last_canvas_item as Control
		if last_control != null:
			last_control.notify_thread_safe(Control.NOTIFICATION_MOUSE_EXIT)
			#last_control.mouse_exited.emit()


func mouse_motion_to_xform(global_pos: Vector2) -> Transform3D:
	#var position2D: Vector2 = get_viewport().get_mouse_position()
	var position2D: Vector2 = global_pos
	var camera: Camera3D = get_viewport().get_camera_3d()
	var ray_origin: Vector3 = camera.project_ray_origin(position2D)
	var ray_normal: Vector3 = camera.project_ray_normal(position2D)

	var dropPlane := camera.global_transform * Plane(Vector3(0, 0, -1), camera.near)
	var position3Dv: Variant = dropPlane.intersects_ray(ray_origin, ray_normal)
	if typeof(position3Dv) != TYPE_VECTOR3:
		print("Fail raycast")
		return Transform3D(Basis.from_scale(Vector3.ZERO), Vector3.ZERO)
	var position3D := position3Dv as Vector3
	return Transform3D(Basis.looking_at(ray_normal), position3D)


# FIXME: We need to allow multiple mouse cursors, which Godot's input system is not designed for.
# So we should ask the caller to pass in the coordinate for each event
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
	#if new_control != null:
	#	var viewport := new_control.get_viewport()
	#	if viewport:
	#		viewport.push_input(ev, true)
	if new_control != null:
		#if false: ####focused_control != new_control:
		#	if focused_control != null:
		#		focused_control.notify_thread_safe(Control.NOTIFICATION_FOCUS_EXIT)
		#		#focused_control.focus_exited.emit()
		#	# new_control.notify_thread_safe(Control.NOTIFICATION_FOCUS_ENTER)
		#	new_control.grab_focus()
		#	new_control.grab_click_focus()
		#	#new_control.focus_entered.emit()
		#	focused_control = new_control
		var viewport = last_canvas_item.get_viewport()
		if viewport:
			viewport.push_input(ev, true)
			#var ev2 := InputEventAction.new()
			#ev2.action = &"ui_accept"
			#ev2.pressed = mb.pressed
			#ev2.event_index = event_index
			#viewport.push_input(ev2)
		#ev.set_global_position(center)
		#ev.set_relative(Vector2(0,0))  # should this be scaled/warped?
		#ev.double_click = mb.double_click
		#ev.factor = mb.factor
		#ev.canceled = mb.canceled
		#ev.button_index = mb.button_index
		#ev.set_button_mask(mb.button_mask)

		#var new_button := last_canvas_item as BaseButton
		#if new_button != null:
			#new_button.button_pressed = ev.pressed
			#new_button.set_pressed_no_signal(ev.pressed)
			#new_button.pressed.emit()
			#new_button.shortcut_input
			#if ev.pressed:
				#print(InputMap.action_get_events(&"ui_accept"))
				#viewport.push_input(InputMap.action_get_events(&"ui_accept")[0])
	#elif focused_control != null:
	#	focused_control.notify_thread_safe(Control.NOTIFICATION_FOCUS_EXIT)
	#	focused_control = null
	#	#focused_control.focus_exited.emit()
