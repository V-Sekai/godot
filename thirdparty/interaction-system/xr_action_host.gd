extends "./action_host.gd"

var xr_tracker: XRControllerTracker

var mb := InputEventMouseButton.new()

func _xr_tracker_button(button_name: StringName, pressed: bool):
	mb.resource_name = button_name
	mb.global_position = Vector2(0,0)
	mb.position = Vector2(0,0)
	mb.canceled = false
	mb.button_index = 1
	mb.button_mask = 1 << mb.button_index
	mb.pressed = pressed
	mb.factor = 1.0 if pressed else 0.0
	mb.double_click = false # FIXME: Calculate by time? Or maybe use a specific binding for double click
	fire_button_event(mb)
	# SteamVR has a double click detection threshold I think
	#interaction_manager.handle_mouse_button(mb)

func _xr_tracker_pose(pose: XRPose):
	fire_pose_changed(pose)
	#var xform := pose.transform
	#interaction_manager.handle_pointer_moved_3d(xform.origin, xform.basis * Vector3(0, 0, -1))

func _xr_pose_lost_tracking(pose: XRPose):
	pose.tracking_confidence = XRPose.XR_TRACKING_CONFIDENCE_NONE
	fire_pose_changed(pose)
