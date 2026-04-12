# spectator_camera.gd
# Fully automated observer camera for monitoring a running FabricZone client.
# Tracks the centroid of `entities_root` children with Arkham-style smoothing,
# auto-fits zoom from the entity bounding sphere, slowly orbits, and cycles
# focus across zones on a fixed wall-clock interval. SpringArm3D handles
# collision push-in. No user input.

extends Node3D

@export var entities_root: NodePath
@export var follow_smoothing: float = 4.0
@export var zoom_smoothing: float = 3.0
@export var orbit_speed: float = 0.08  # rad/sec — slow FFT-style rotation
@export var pitch: float = -1.05  # ~60° from horizontal — FFT isometric
@export var fit_padding: float = 1.6
@export var min_distance: float = 4.0
@export var max_distance: float = 60.0
@export var zone_count: int = 3
@export var zone_extent: float = 15.0
@export var cycle_interval: float = 8.0  # seconds per zone; 0 disables cycling

var _yaw: float = 0.0
var _smoothed_position: Vector3 = Vector3.ZERO
var _smoothed_distance: float = 12.0
var _spring: SpringArm3D
var _entities: Node3D
var _cycle_timer: float = 0.0
var _focus_index: int = -1  # -1 = whole-world centroid, 0..zone_count-1 = zone

func _ready() -> void:
	_spring = $SpringArm3D
	_spring.spring_length = _smoothed_distance
	if entities_root != NodePath():
		_entities = get_node_or_null(entities_root) as Node3D

func _process(delta: float) -> void:
	if cycle_interval > 0.0:
		_cycle_timer += delta
		if _cycle_timer >= cycle_interval:
			_cycle_timer = 0.0
			# Cycle: -1 (whole world) → 0 → 1 → ... → zone_count-1 → -1 → ...
			_focus_index = -1 if _focus_index >= zone_count - 1 else _focus_index + 1

	var goal_pos := _focus_position()
	var goal_radius := _focus_radius(goal_pos)

	_smoothed_position = _smoothed_position.lerp(goal_pos, clampf(follow_smoothing * delta, 0.0, 1.0))
	position = _smoothed_position

	var goal_dist: float = clampf(goal_radius * fit_padding, min_distance, max_distance)
	_smoothed_distance = lerpf(_smoothed_distance, goal_dist, clampf(zoom_smoothing * delta, 0.0, 1.0))
	_spring.spring_length = _smoothed_distance

	_yaw += orbit_speed * delta
	rotation = Vector3(pitch, _yaw, 0.0)

func _focus_position() -> Vector3:
	if _focus_index >= 0:
		return _zone_centroid(_focus_index)
	if _entities == null or _entities.get_child_count() == 0:
		return Vector3.ZERO
	var sum := Vector3.ZERO
	var n := 0
	for child in _entities.get_children():
		var n3 := child as Node3D
		if n3 != null:
			sum += n3.global_position
			n += 1
	return sum / float(n) if n > 0 else Vector3.ZERO

func _focus_radius(center: Vector3) -> float:
	if _focus_index >= 0:
		return 2.0 * zone_extent / float(zone_count)
	if _entities == null or _entities.get_child_count() == 0:
		return min_distance
	var max_sq := 0.0
	for child in _entities.get_children():
		var n3 := child as Node3D
		if n3 != null:
			var d := center.distance_squared_to(n3.global_position)
			if d > max_sq:
				max_sq = d
	return sqrt(max_sq)

func _zone_centroid(idx: int) -> Vector3:
	var slab := 2.0 * zone_extent / float(zone_count)
	var z := -zone_extent + slab * (float(idx) + 0.5)
	return Vector3(0.0, 0.0, z)
