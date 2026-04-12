extends Node3D

const interaction_manager_class := preload("../interaction_manager.gd")

var interaction_manager: interaction_manager_class

var enabled: bool = true
var primary_pose_name: StringName = &"aim"

func on_action_added() -> void:
	pass

func on_action_removed() -> void:
	pass

func on_button_event(mb: InputEventMouseButton) -> bool:
	return false

func on_pose_changed(pose: XRPose):
	if pose.tracking_confidence == XRPose.XR_TRACKING_CONFIDENCE_NONE:
		return
	if pose.name == primary_pose_name:
		transform = pose.transform
