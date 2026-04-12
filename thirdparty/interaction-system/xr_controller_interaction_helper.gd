extends Node3D

const interaction_manager_class := preload("./interaction_manager.gd")
const xr_action_host_class := preload("./xr_action_host.gd")

var interaction_manager: interaction_manager_class

var trackers: Dictionary[StringName, xr_action_host_class]

@export var controller_scene: PackedScene

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	interaction_manager = get_viewport().get_meta(&"interaction_manager")

	for tracker_key in XRServer.get_trackers(XRServer.TRACKER_CONTROLLER).keys():
		_tracker_added(tracker_key as StringName, 0)
	XRServer.tracker_added.connect(_tracker_added)
	XRServer.tracker_removed.connect(_tracker_removed)


func _tracker_added(tracker_name: StringName, type: int):
	var tracker: XRControllerTracker = XRServer.get_tracker(tracker_name) as XRControllerTracker
	# Note: XRHandTracker is not XRControllerTracker
	if tracker == null or tracker.type != XRServer.TRACKER_CONTROLLER:
		return
	var xr_action_host: xr_action_host_class
	if trackers.has(tracker_name):
		xr_action_host = trackers[tracker_name]
		if xr_action_host.visible:
			push_error("Duplicate _tracker_added xr_action_host Node for " + str(tracker_name))
		xr_action_host.visible = true
	else:
		var tmp: Node3D = controller_scene.instantiate()
		tmp.name = tracker_name
		if not (tmp is xr_action_host_class):
			tmp.set_script(xr_action_host_class)
		xr_action_host = tmp as xr_action_host_class
		xr_action_host.interaction_manager = interaction_manager
		xr_action_host.xr_tracker = tracker
		trackers[tracker_name] = xr_action_host
		add_child(xr_action_host)
	interaction_manager.register_action_host(xr_action_host)
	tracker.button_pressed.connect(xr_action_host._xr_tracker_button.bind(true))
	tracker.button_released.connect(xr_action_host._xr_tracker_button.bind(false))
	tracker.pose_changed.connect(xr_action_host._xr_tracker_pose)
	tracker.pose_lost_tracking.connect(xr_action_host._xr_pose_lost_tracking)
	# TODO: figure out scrolling

func _tracker_removed(tracker_name: StringName, type: int):
	var tracker: XRControllerTracker = XRServer.get_tracker(tracker_name) as XRControllerTracker
	if tracker == null or tracker.type != XRServer.TRACKER_CONTROLLER:
		return
	if not trackers.has(tracker_name):
		push_error("_tracker_removed for unknown tracker " + str(tracker_name))
	var xr_action_host := trackers[tracker_name]
	if not xr_action_host.visible:
		push_error("_tracker_removed without _tracker_added for " + str(tracker_name))
		xr_action_host.visible = false
	tracker.button_pressed.disconnect(xr_action_host._xr_tracker_button.bind(true))
	tracker.button_released.disconnect(xr_action_host._xr_tracker_button.bind(false))
	tracker.pose_changed.disconnect(xr_action_host._xr_tracker_pose)
	tracker.pose_lost_tracking.disconnect(xr_action_host._xr_pose_lost_tracking)
	interaction_manager.unregister_action_host(xr_action_host)
