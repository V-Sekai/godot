@uid("uid://dmisgwrk2os7b") # Generated automatically, do not modify.
@tool
extends EditorPlugin

var bottompanel: Control

const MMEditorGizmoPlugin = preload("res://addons/MM/MMEditorGizmoPlugin.gd")

var gizmo_plugin := MMEditorGizmoPlugin.new()


func _enter_tree() -> void:
	add_node_3d_gizmo_plugin(gizmo_plugin)
	bottompanel = preload("res://addons/MM/MMEditorPanel.tscn").instantiate()
	get_editor_interface().get_selection().selection_changed.connect(visibility)
	add_control_to_bottom_panel(bottompanel, "Motion Matching")


func _exit_tree() -> void:
	remove_node_3d_gizmo_plugin(gizmo_plugin)
	remove_control_from_bottom_panel(bottompanel)
	if(bottompanel != null):
		bottompanel.free()


func _has_main_screen()->bool:
	return false


func visibility() -> void:
	var nodes :Array= get_editor_interface().get_selection().get_selected_nodes()
	var v = nodes.any(func(x):return x is MotionPlayer)
	bottompanel.visible = v

	if v:
		print(get_tree())
		bottompanel._current = nodes.filter(func (x): return x is MotionPlayer)[0]
		bottompanel._animplayer = bottompanel._current.owner.find_children("*","AnimationPlayer",true,true)[0]
		add_control_to_bottom_panel(bottompanel,"MotionMatching")
		make_bottom_panel_item_visible(bottompanel)
	else :
		remove_control_from_bottom_panel(bottompanel)
