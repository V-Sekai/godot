extends Node

var _mouse_controller: XRControllerTracker = XRControllerTracker.new()

var _current_camera: Camera3D

func _enter_tree() -> void:
	_mouse_controller.name = &"/mouse"
	_mouse_controller.type = XRServer.TRACKER_CONTROLLER
	_mouse_controller.profile = "mouse"
	_mouse_controller.description = "Virtual controller for mouse input"
	XRServer.add_tracker(_mouse_controller)

func _exit_tree() -> void:
	XRServer.remove_tracker(_mouse_controller)

func _process(delta: float) -> void:
	_current_camera = get_viewport().get_camera_3d()

func _input(event: InputEvent):
	var me := event as InputEventMouseMotion
	if me != null:
		handle_mouse_motion(me.global_position)
	var mb := event as InputEventMouseButton
	if mb != null:
		handle_mouse_button(mb)

func handle_mouse_motion(global_pos: Vector2):
	if _current_camera == null:
		print("camera is null")
		return
	#var position2D: Vector2 = get_viewport().get_mouse_position()
	var position2D: Vector2 = global_pos
	var ray_origin: Vector3 = _current_camera.project_ray_origin(position2D)
	var ray_normal: Vector3 = _current_camera.project_ray_normal(position2D)
	
	var plane_normal: Vector3 = Vector3(0, 0, -1)

	var center_transform: Transform3D = _current_camera.global_transform
	var dropPlane := center_transform * Plane(Vector3(0, 0, -1), _current_camera.near)
	var position3Dv: Variant = dropPlane.intersects_ray(ray_origin, ray_normal)
	if typeof(position3Dv) != TYPE_VECTOR3:
		print("Fail raycast")
		return
	var position3D := position3Dv as Vector3

	#var from_to_basis := center_transform.basis * Basis(Quaternion(center_transform.basis * plane_normal, ray_normal))
	var right_vec := Vector3(0,1,0).cross(ray_normal)
	var up_vec := ray_normal.cross(right_vec)
	var from_to_basis := Basis(right_vec, up_vec, -ray_normal)

	_mouse_controller.set_pose(&"aim", Transform3D(from_to_basis, position3D) * Transform3D(Basis.IDENTITY, Vector3(-0.005, -.01, -0.1)), Vector3.ZERO, Vector3.ZERO, XRPose.XR_TRACKING_CONFIDENCE_HIGH)
	## (camera.global_transform, camera.near, ray_origin, ray_normal)


func handle_mouse_button(mb: InputEventMouseButton):
	handle_mouse_motion(mb.global_position)
	var float_pressed := mb.factor * (1.0 if mb.pressed else 0.0)
	var input_name := &""
	var input_value: Variant = null
	
	match mb.button_index:
		MouseButton.MOUSE_BUTTON_LEFT:
			input_name = &"left_click"
			input_value = mb.pressed
		MouseButton.MOUSE_BUTTON_RIGHT:
			input_name = &"right_click"
			input_value = mb.pressed
		MouseButton.MOUSE_BUTTON_MIDDLE:
			input_name = &"middle_click"
			input_value = mb.pressed
		MouseButton.MOUSE_BUTTON_WHEEL_UP:
			input_name = &"scroll"
			input_value = Vector2(0, float_pressed)
		MouseButton.MOUSE_BUTTON_WHEEL_DOWN:
			input_name = &"scroll"
			input_value = Vector2(0, -float_pressed)
		MouseButton.MOUSE_BUTTON_WHEEL_LEFT:
			input_name = &"scroll"
			input_value = Vector2(float_pressed, 0)
		MouseButton.MOUSE_BUTTON_WHEEL_RIGHT:
			input_name = &"scroll"
			input_value = Vector2(-float_pressed, 0)
		MouseButton.MOUSE_BUTTON_XBUTTON1:
			input_name = &"extra1"
			input_value = mb.pressed
		MouseButton.MOUSE_BUTTON_XBUTTON2:
			input_name = &"extra2"
			input_value = mb.pressed

	_mouse_controller.set_input(input_name, input_value)
