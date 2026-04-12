extends "./base_action.gd"

@export_node_path("MeshInstance3D") var laser_mesh: NodePath = ^"Laser":
	set(n):
		laser_mesh = n
		laser_mesh_instance = get_node_or_null(n) as MeshInstance3D
@onready var laser_mesh_instance: MeshInstance3D = get_node_or_null(laser_mesh) as MeshInstance3D

var query := interaction_manager_class.LassoQuery.new()

var current_poi: interaction_manager_class.LassoPOI
var current_canvas_item: CanvasItem
var current_canvas_plane: interaction_manager_class.canvas_plane_class
var current_pos_2d: Vector2
var current_target_pos_3d: Vector3
var pressed_buttons: Dictionary[StringName, bool]

func on_action_added() -> void:
	pass

func on_action_removed() -> void:
	pass

func on_button_event(mb: InputEventMouseButton) -> bool:
	var xform := transform
	if mb.pressed:
		pressed_buttons[mb.resource_name] = true
	else:
		pressed_buttons.erase(mb.resource_name)
	interaction_manager.handle_pointer_moved_2d(null, current_canvas_item, current_pos_2d)
	mb.global_position = current_pos_2d.round()
	mb.position = current_pos_2d.round()
	interaction_manager.handle_mouse_button(current_canvas_item, mb)
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
	interaction_manager.handle_pointer_moved_2d(old_canvas_item, current_canvas_item, current_pos_2d)
	if laser_mesh_instance != null:
		# print("Target pos 3d = " + str(current_target_pos_3d) + " / Target 2d = " + str(current_pos_2d))
		laser_mesh_instance.set_instance_shader_parameter(&"target", laser_mesh_instance.global_transform.affine_inverse() * current_target_pos_3d)
