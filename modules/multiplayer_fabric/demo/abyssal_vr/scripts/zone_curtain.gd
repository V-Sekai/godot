# zone_curtain.gd
# Renders the Hilbert zone boundary as a staircase curtain.
#
# Build phase: spawn per-face CSGBox3D slabs (manifold, SHROUD_THICKNESS deep)
#   under a CSGCombiner3D so CSG union merges coplanar faces.
# Bake phase (next frame): call get_meshes() on the combiner, create a
#   MeshInstance3D from the result, then free all CSG nodes.
# Runtime: pulse emission_energy_multiplier on the baked mesh's material.

extends Node3D

const SIM_BOUND        := 15.0   # meters — matches FabricZone::SIM_BOUND
const SAMPLE_DEPTH     := 4      # cells per axis  (4^3 = 64 total)
const SHROUD_THICKNESS := 0.05   # meters — slab depth (non-zero → manifold)
@export var zone_count: int = 3

const ZONE_COLORS: Array[Color] = [
	Color(0.9, 0.3, 0.3, 0.18),   # zone 0 — warm red
	Color(0.3, 0.9, 0.5, 0.18),   # zone 1 — aqua green
	Color(0.3, 0.5, 1.0, 0.18),   # zone 2 — deep blue
]

var _mesh_instances: Array[MeshInstance3D] = []
var _time: float = 0.0


func _ready() -> void:
	if zone_count < 1:
		return
	if zone_count >= 2:
		_build_csg_then_bake()
	_spawn_zone_labels()


func _spawn_zone_labels() -> void:
	var depth := _prefix_depth(zone_count)
	var cell_w := 1 << (30 - depth) if depth < 30 else 1
	for zi in range(zone_count):
		var zone_lo := zi * cell_w
		var cell_aabb := FabricZone.hilbert_cell_of_aabb(zone_lo, depth)
		var centroid := cell_aabb.get_center()
		centroid.y = 0.5
		var label := Label3D.new()
		label.text = "ZONE %d" % zi
		label.billboard = BaseMaterial3D.BILLBOARD_DISABLED
		label.no_depth_test = true
		label.fixed_size = false
		label.pixel_size = 0.03
		label.font_size = 64
		label.outline_size = 16
		label.modulate = ZONE_COLORS[zi % ZONE_COLORS.size()]
		label.modulate.a = 1.0
		label.outline_modulate = Color.BLACK
		label.position = centroid
		label.rotation_degrees = Vector3(-90, 0, 0)
		add_child(label)


func _process(delta: float) -> void:
	_time += delta
	var pulse := 0.4 + 0.4 * sin(_time * 2.0)
	for mi in _mesh_instances:
		var mat := mi.get_active_material(0) as StandardMaterial3D
		if mat:
			mat.emission_energy_multiplier = pulse


# ── Build ─────────────────────────────────────────────────────────────────────

func _build_csg_then_bake() -> void:
	var n := SAMPLE_DEPTH
	var cell_size := (SIM_BOUND * 2.0) / float(n)

	# Zone per cell
	var cells := PackedInt32Array()
	cells.resize(n * n * n)
	for ix in range(n):
		for iy in range(n):
			for iz in range(n):
				cells[_idx(ix, iy, iz, n)] = _zone_for_point(
					-SIM_BOUND + (ix + 0.5) * cell_size,
					-SIM_BOUND + (iy + 0.5) * cell_size,
					-SIM_BOUND + (iz + 0.5) * cell_size)

	# One CSGCombiner3D per zone — keeps zone colors separate after bake.
	var combiners: Array[CSGCombiner3D] = []
	for _z in range(zone_count):
		var c := CSGCombiner3D.new()
		c.use_collision = false
		add_child(c)
		combiners.append(c)

	# Spawn per-face slabs into the combiner of the lower-index zone.
	# Check only +x/+y/+z to avoid duplicate slabs.
	for ix in range(n):
		for iy in range(n):
			for iz in range(n):
				var z0: int = cells[_idx(ix, iy, iz, n)]
				_check_face(combiners, cells, ix, iy, iz, n, cell_size,
					ix + 1, iy, iz,   Vector3(SHROUD_THICKNESS, cell_size, cell_size),
					-SIM_BOUND + (ix + 1) * cell_size,
					-SIM_BOUND + (iy + 0.5) * cell_size,
					-SIM_BOUND + (iz + 0.5) * cell_size, z0)
				_check_face(combiners, cells, ix, iy, iz, n, cell_size,
					ix, iy + 1, iz,   Vector3(cell_size, SHROUD_THICKNESS, cell_size),
					-SIM_BOUND + (ix + 0.5) * cell_size,
					-SIM_BOUND + (iy + 1) * cell_size,
					-SIM_BOUND + (iz + 0.5) * cell_size, z0)
				_check_face(combiners, cells, ix, iy, iz, n, cell_size,
					ix, iy, iz + 1,   Vector3(cell_size, cell_size, SHROUD_THICKNESS),
					-SIM_BOUND + (ix + 0.5) * cell_size,
					-SIM_BOUND + (iy + 0.5) * cell_size,
					-SIM_BOUND + (iz + 1) * cell_size,   z0)

	# CSGShape3D.bake_static_mesh() synchronously returns the unioned ArrayMesh,
	# bypassing the deferred get_meshes() path entirely.
	_bake(combiners)


func _check_face(
		combiners: Array[CSGCombiner3D],
		cells: PackedInt32Array,
		ix: int, iy: int, iz: int, n: int, _cell_size: float,
		nx: int, ny: int, nz: int,
		slab_size: Vector3, px: float, py: float, pz: float,
		z0: int) -> void:
	if nx >= n or ny >= n or nz >= n:
		return
	var z1: int = cells[_idx(nx, ny, nz, n)]
	if z1 == z0:
		return
	var box := CSGBox3D.new()
	box.size = slab_size
	box.position = Vector3(px, py, pz)
	box.use_collision = false
	combiners[min(z0, z1)].add_child(box)


# ── Bake ──────────────────────────────────────────────────────────────────────

func _bake(combiners: Array[CSGCombiner3D]) -> void:
	for zi in range(combiners.size()):
		var combiner := combiners[zi]
		combiner._update_shape()   # Force synchronous CSG union (deferred otherwise).
		var mesh := combiner.bake_static_mesh()
		print("[zone_curtain] zone=%d bake surfaces=%d" % [zi, 0 if mesh == null else mesh.get_surface_count()])
		if mesh != null and mesh.get_surface_count() > 0:
			var mi := MeshInstance3D.new()
			mi.mesh = mesh
			mi.material_override = _make_material(zi)
			add_child(mi)
			_mesh_instances.append(mi)
		combiner.queue_free()


func _make_material(zone_idx: int) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	var col := ZONE_COLORS[zone_idx % ZONE_COLORS.size()]
	mat.albedo_color = col
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mat.emission_enabled = true
	mat.emission = col
	mat.emission_energy_multiplier = 0.4
	return mat


# ── Hilbert helpers ───────────────────────────────────────────────────────────

func _zone_for_point(cx: float, cy: float, cz: float) -> int:
	return _zone_for_hilbert(FabricZone.hilbert_of_point(cx, cy, cz), zone_count)


# Matches Fabric.lean shardPrefixDepth / assignToShard:
#   depth = ceil(log2(count)) = bit-length of (count - 1)
#   zone  = min(count - 1, code >>> (30 - depth))
func _prefix_depth(count: int) -> int:
	if count <= 1:
		return 0
	var x := count - 1
	var depth := 0
	while x > 0:
		depth += 1
		x >>= 1
	return depth


func _zone_for_hilbert(code: int, count: int) -> int:
	if count <= 1:
		return 0
	var depth := _prefix_depth(count)
	return min(count - 1, code >> (30 - depth))


func _idx(ix: int, iy: int, iz: int, n: int) -> int:
	return ix * n * n + iy * n + iz
