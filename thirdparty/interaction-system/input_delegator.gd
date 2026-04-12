extends Node

const interaction_manager_class := preload("./interaction_manager.gd")

var focused_interaction_manager: interaction_manager_class


# sources: mouse (input) / keyboard, VR aim, VR joystick?
# sink: focused interaction manager?
# sink: custom modules?

# pluggable input? allow custom interaction manager to sit in between?

# while interacting with object, it becomes modal (this is how we can open the context menu
# for an item you click: while holding it, other options appear, radial menu style?
# or, click and release to keep menu open, and we need a close button or some way out of modal.

# ifire: Resoneos style tooltips? this is what I was referring to as custom modules maybe?
# or maybe it's something like the modal interaction with the tooltip starts and never ends
# and now everything goes through the tooltip modal, but instead of having its own interaction manager
# and viewport, it uses the same viewport as the default.
# Tooltip = "Lens". idea of augmentation


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass


func _input(event: InputEvent):
	# if event is InputEventMouseMotion:
	#	if focused_interaction_manager != null:
	#		focused_interaction_manager.handle_input(event)
	var me := event as InputEventMouseMotion
	if me != null:
		if focused_interaction_manager != null:
			focused_interaction_manager.handle_mouse_motion(me.global_position)
	var mb := event as InputEventMouseButton
	if mb != null:
			focused_interaction_manager.handle_mouse_button(mb)
	#	var center = last_coord # last_canvas_item.get_global_transform().origin
	#	var ev := InputEventMouseButton.new()
	#	var new_button := last_canvas_item as BaseButton
