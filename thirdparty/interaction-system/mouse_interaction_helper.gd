extends "./action_host.gd"

var xr_pose := XRPose.new()

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	interaction_manager = get_viewport().get_meta(&"interaction_manager")
	interaction_manager.register_action_host(self)

func _exit_tree() -> void:
	if interaction_manager != null:
		interaction_manager.unregister_action_host(self)

func _input(event: InputEvent):
	var me := event as InputEventMouseMotion
	if me != null:
		xr_pose.name = &"aim"
		xr_pose.tracking_confidence = XRPose.XR_TRACKING_CONFIDENCE_HIGH
		var xform: Transform3D = interaction_manager.mouse_motion_to_xform(me.global_position)
		xr_pose.transform = xform
		fire_pose_changed(xr_pose)
		#interaction_manager.handle_pointer_moved_3d(xform.origin, xform.basis * Vector3(0, 0, -1))
	var mb := event as InputEventMouseButton
	if mb != null:
		var mb_clone = mb.clone()
		mb_clone.resource_name = str(mb.button_index)
		fire_button_event(mb_clone)
		#interaction_manager.handle_mouse_button(mb)
