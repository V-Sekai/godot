extends RefCounted

class PointOfInterest:
	extends RefCounted
	@export var snapping_power: float = 1.0
	@export var snapping_enabled: bool = true
	@export var snap_locked: bool = true
	@export var size: float = 0.3
	var last_snap_score: float = 0.0
	var db: WeakRef # LassoDB. Avoid reference cycle.
	var origin: Node3D

	func register_point(db_: RefCounted, origin_: Node):
		origin = origin_ as Node3D
		if not origin:
			return
		db = weakref(db_) as WeakRef
		db_.add_point(self)

	func unregister_point():
		var db_: RefCounted = db.get_ref()
		if db_:
			db_.remove_point(self)

	func get_origin_pos() -> Vector3:
		return origin.global_position

	func get_origin_transformed_pos(source: Transform3D) -> Vector3:
		#var local_source = source.origin * origin.global_transform
		var pos: Vector3 = origin.global_position
		# var canvas_item := origin.get(&"canvas_item") as CanvasItem
		var aabb: AABB
		if origin.has_method(&"get_aabb"):
			aabb = origin.get_aabb() as AABB
			#if not aabb.get_center().is_zero_approx():
			#	pos += origin.global_basis * aabb.get_center()
		#if canvas_item != null:
		#	# store local variable to workaround missing vec2->vec3 conversion
		#	var dims_vec2 := canvas_item.get_viewport_rect().size
		#	dims = Vector3(dims_vec2.x, dims_vec2.y, 0.0)
		var source_pos: Vector3 = source.affine_inverse() * pos
		if aabb.size.is_zero_approx():
			return source_pos

		# TODO: Switch to using AABB instead of local_pos.clamp()
		# THEN ... we can use
		# var ray_intersect = aabb.intersects_ray(source.origin * origin.global_transform, origin.global_basis * Vector3(0,0,-1))


		var closest_box_point: Vector3
		#var local_pos: Vector3 = (source * Vector3(0, 0, source_pos.z)) * origin.global_transform # inverse??
		#print(local_pos)
		#var closest_box_point := local_pos.clamp(-0.5 * dims, 0.5 * dims)
		var closest_box_point_var: Variant = AABB(aabb.position * Vector3(1000,1000,1000), aabb.size * Vector3(1000,1000,1000)).intersects_ray(
			origin.global_transform.affine_inverse() * source.origin,
			((source.basis * Vector3.FORWARD) * origin.global_basis).normalized())
		if typeof(closest_box_point_var) == TYPE_VECTOR3:
			closest_box_point = closest_box_point_var as Vector3
		else:
			closest_box_point = origin.global_transform.affine_inverse() * (source * Vector3(0, 0, source_pos.z))
		closest_box_point = closest_box_point.clamp(aabb.position, aabb.size + aabb.position)

		# pos += origin.global_basis * closest_box_point
		pos = origin.global_transform * closest_box_point
		# print("GLOBAL POS : " + str(pos))

		#var dim_matrix = Basis.from_scale(dims)
		#var source_dim_matrix: Basis = (origin.global_basis * dim_matrix) * source.basis
		#source_dim_matrix 
		#var z_axis = (origin.global_basis.inverse() * source.basis * Vector3.BACK).normalized()
		return source.affine_inverse() * pos

	func BORKEN_get_origin_pos(source: Transform3D) -> Vector3:
		# Botched attempt.
		
		# The issue is there is a position check AND an angle check... we really want to fix the angle chec
		# You can't fix the source position since even a tiny offset would mess up all the math
		# so we need to fix the angle and assist that check...
		'''
		var origin_width: float = origin.get_meta("width") or 0
		var origin_height: float = origin.get_meta("height") or 0
		if origin_width > 0 or origin_height > 0:
			var source_in_local: Vector3 = source.origin * origin.global_transform
			var subtract_axis: Vector3
			if source_in_local.x > origin_width - origin_height:
				source_in_local.x -= origin_width - origin_height
			if source_in_local.x < -(origin_width - origin_height):
				source_in_local.x += origin_width - origin_height
			if source_in_local.y > origin_width - origin_height:
				source_in_local.x -= origin_width - origin_height
			if source_in_local.x < -(origin_width - origin_height):
				source_in_local.x += origin_width - origin_height
				subtract_axis = Vector3(origin_width - origin_height, 0, 0)
			else:
				subtract_axis = Vector3(0, origin_height - origin_width, 0)
		var ret = origin.global_position
		'''
		return Vector3.ZERO

	func is_valid_origin() -> bool:
		return origin != null and origin.is_visible_in_tree() and snapping_enabled

	func is_matching_origin(p_origin: Node) -> bool:
		return origin == p_origin

# var points: Array[PointOfInterest]
#
var point_set: Dictionary[PointOfInterest, bool]

func add_point(point: PointOfInterest) -> void:
	if point and not point_set.has(point):
		# points.append(point)
		point_set[point] = true

func remove_point(point: PointOfInterest) -> void:
	# var idx: int = points.find(point)
	# if idx >= 0: points.remove_at(idx)
	point_set.erase(point)

class LassoQuery extends RefCounted:
	var source: Transform3D
	var override_point_set: Dictionary[PointOfInterest, bool]
	var current_snap: PointOfInterest
	var snap_max_power_increase: float
	var snap_increase_amount: float
	var snap_lock: bool
	var min_snap_score: float
	
	func set_source(position3D: Vector3, ray_normal: Vector3):
		source = Transform3D(Basis.looking_at(ray_normal), position3D)

	func get_position_3d(poi: PointOfInterest) -> Vector3:
		if not out_poi_to_local.has(poi):
			return source.origin
		return source * out_poi_to_local[poi]

	var out_best_poi: PointOfInterest
	var out_poi_to_local: Dictionary[PointOfInterest, Vector3]

var tmp_query := LassoQuery.new()

func calc_top_two_snapping_power(source: Transform3D, current_snap: PointOfInterest,
		snap_max_power_increase: float, snap_increase_amount: float, snap_lock: bool,
		min_snap_score: float) -> Array:
	tmp_query.source = source
	tmp_query.current_snap = current_snap
	tmp_query.snap_max_power_increase = snap_max_power_increase
	tmp_query.snap_increase_amount = snap_increase_amount
	tmp_query.snap_lock = snap_lock
	tmp_query.min_snap_score = min_snap_score

	var first: PointOfInterest = tmp_query.out_best_poi
	var second: PointOfInterest = null
	for poi in tmp_query.out_poi_to_local:
		if poi != first:
			second = poi
			break
	var output := []
	output.push_back(first)
	output.push_back(second)
	if first == null:
		output.push_back(source.origin)
	else:
		output.push_back(source * tmp_query.out_poi_to_local[first])
	if second == null:
		output.push_back(source.origin)
	else:
		output.push_back(source * tmp_query.out_poi_to_local[second])
	return output


func query(query: LassoQuery) -> bool:
	var first: PointOfInterest
	var first_local: Vector3
	var second: PointOfInterest
	var second_local: Vector3
	var current_point_set := query.override_point_set
	if current_point_set.is_empty():
		current_point_set = point_set
	for pt in current_point_set:
		var next := pt as PointOfInterest
		if next and next.is_valid_origin() and next.snapping_enabled:
			var point_local: Vector3 = next.get_origin_transformed_pos(query.source)
			var euclidian_dist: float = point_local.length()
			var angular_dist: float = point_local.angle_to(Vector3(0, 0, -1))
			var rejection_length: float = point_local.length() # Vector3(point_local[0], point_local[1], 0).length()

			var snapping_power: float = 0
			if rejection_length <= next.size:
				snapping_power = next.snapping_power / (1.0 + euclidian_dist) / (0.01 + angular_dist)
			else:
				snapping_power = next.snapping_power / (1.0 + euclidian_dist) / (0.1 + angular_dist)

			if next == query.current_snap: # next.is_matching_origin(query.current_snap.origin):
				snapping_power += snapping_power * pow(query.snap_increase_amount, 2) * query.snap_max_power_increase
				if next.snap_locked and query.snap_lock:
					next.last_snap_score = snapping_power
					first = next;
					first_local = point_local
					second = null
					break

			next.last_snap_score = snapping_power
			if next.last_snap_score < query.min_snap_score:
				continue

			if first == null or first.last_snap_score < next.last_snap_score:
				second = first;
				second_local = first_local
				first = next;
				first_local = point_local
			elif second == null || second.last_snap_score < next.last_snap_score:
				second = next
				second_local = point_local
	query.out_best_poi = first
	if first != null:
		query.out_poi_to_local[first] = first_local
	if second != null:
		query.out_poi_to_local[second] = second_local
	return first != null


func calc_top_redirecting_power(snapped_origin: PointOfInterest,
		viewpoint: Transform3D,
		redirection_direction: Vector2) -> PointOfInterest:
	var snapped_origin_Node3D := snapped_origin.origin as Node3D
	if snapped_origin_Node3D == null:
		return null

	var output: PointOfInterest = snapped_origin

	if not redirection_direction.is_zero_approx():
		# Caclculate the basis.
		var snapped_origin_position: Vector3 = snapped_origin_Node3D.global_position;
		var snapped_vector: Vector3 = viewpoint.origin - snapped_origin_position;
		var z_vector: Vector3 = snapped_vector.normalized()
		var up_vector: Vector3 = viewpoint.basis.y.normalized()
		var x_vector: Vector3 = z_vector.cross(up_vector).normalized()
		var y_vector: Vector3 = x_vector.cross(z_vector).normalized()
		var local_basis := Basis(x_vector, y_vector, z_vector)
		var first: PointOfInterest
		var redirect_power: float = INF # The lower is better.
		for pt in point_set:
			var next := pt as PointOfInterest
			var next_power: float = 0
			if next and next.is_valid_origin() and not next.is_matching_origin(snapped_origin_Node3D):
				var point_vector: Vector3 = viewpoint.origin - next.get_origin_pos()
				if point_vector.angle_to(snapped_vector) < PI / 4.0:
					var point_xyz: Vector3 = local_basis * point_vector
					var point_xy := Vector2(point_xyz[0], -point_xyz[1])

					if absf(redirection_direction.angle_to(point_xy)) >= PI / 2:
						continue
						# Keep the redirect power at infinity if the joystick is more than
						# 90 degrees away from the point.
					elif redirection_direction[0] == 0 && point_xy[1] != 0:
						# If you moved your joystick perfectly vertically calculating the
						# intersection of two lines breaks because of the y = mx + b
						# notation so instead let's calculate the y intercept of the line.
						var bisecting_slope: float = -point_xy[0] / point_xy[1] # Rotated 90 because it's the bisecting line.
						var bisecting_y: float = point_xy[1] / 2 # Y value when the bisecting line intersects the line to the point.
						# Squared just because. we're not actually calculating dist.
						next_power = pow(bisecting_slope * -(point_xy[0] / 2) + bisecting_y, 2)
					elif point_xy[1] == 0:
						# Point is on the x-axis which means the bisecting line would be
						# vertical and also undefined we calculate.
						var bisecting_x: float = point_xy[0] / 2;
						var slope: float = redirection_direction[1] / redirection_direction[0] # rise over run
						var intersect_x: float = bisecting_x;
						var intersect_y: float = bisecting_x * slope;
						next_power = pow(intersect_x, 2) + pow(intersect_y, 2) # squared euclidean distance
					else:
						# This is the most common case
						# equation taken from the internet.
						var a1: float = -point_xy[0] / point_xy[1]
						var c1: float = (1 - a1) * point_xy[0] / 2
						var a2: float = redirection_direction[1] / redirection_direction[0]
						var x_component: float = c1 / (a2 - a1)
						var y_component: float = (a2 * c1) / (a2 - a1)
						next_power = pow(x_component, 2) + pow(y_component, 2);

					if next_power < redirect_power:
						redirect_power = next_power;
						first = next;

		if first != null and first.is_valid_origin():
			output = first

	return output
