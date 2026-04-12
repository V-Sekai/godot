# fabric_client.gd
# Connects to a FabricZone as a player, decodes CH_INTEREST entity snapshots,
# and maintains a pool of visual nodes (JellyfishVisual / WhaleVisual).
#
# CH_INTEREST packet format (100 bytes per entity):
#   [u32 gid(4)][f64 cx(8)][f64 cy(8)][f64 cz(8)]
#   [i16 vx(2)][i16 vy(2)][i16 vz(2)][i16 ax(2)][i16 ay(2)][i16 az(2)]
#   [u32 hlc(4)][u32×14 payload(56)]
#   hlc = tick(24b) | counter(8b)
# Broadcast every 5 physics ticks, chunked at 1200 bytes (~12 entities).
#
# Entity classification by global_id:
#   0–255   → JellyfishVisual (jellyfish_bloom)
#   256–399 → JellyfishVisual (jellyfish_zone_crossing)
#   400–511 → WhaleVisual     (global_id % 14 == 0 = whale, else remora)

extends Node

const CH_INTEREST           := 1
const PACKET_ENTRY_SIZE     := 100  # bytes per entity in CH_INTEREST:
                                    # 4(gid)+3×8(xyz)+6×2(vel/accel)+4(hlc)+14×4(payload) = 100
const PAYLOAD_OFFSET        := 44   # byte offset of payload[0] within each entry (unchanged)
const V_MAX_PHYSICAL        := 500_000.0  # μm/tick — mirror of PBVH_V_MAX_PHYSICAL_DEFAULT (Lean-derived at the server's default tick rate)
const V_SCALE               := 32767.0 / (V_MAX_PHYSICAL * 1.0e-6)  # int16 → m/tick
const A_SCALE               := 32767.0 / (2.0 * V_MAX_PHYSICAL * 1.0e-6)  # int16 → m/tick²
const ENTITY_TIMEOUT_FRAMES := 60   # despawn after missed interest updates
const STROKE_ENTITY_BASE    := 1000000

@export var zone_host: String = "127.0.0.1"
@export var zone_port: int = 17500
@export var entities_root: NodePath
@export var xr_origin: NodePath = NodePath("../XROrigin3D/XRCamera3D")
# player_id derives from the listen port offset so each of the three smoke
# clients picks a unique id. Override per-instance from the command line.
@export var player_id: int = 0

var _peer: FabricMultiplayerPeer
var _entity_nodes: Dictionary = {}   # {int global_id -> Node3D}
var _entity_last_seen: Dictionary = {} # {int global_id -> int frame_count}
var _frame_count: int = 0

# Phase-1 pass condition: track zone-crossing entities (256-399) seen and snap events.
const XING_ID_LO := 256
const XING_ID_HI := 399
const XING_TOTAL := 144
const SNAP_THRESHOLD_M := 1.0  # meters: position jump larger than this = snap
var _xing_seen: Dictionary = {}      # {gid -> bool} zone-crossing entities ever seen
var _xing_positions: Dictionary = {} # {gid -> Vector3} last known position for snap detection
var _snap_count: int = 0
var _pass_logged: bool = false

@onready var _root: Node3D = get_node(entities_root)
@onready var _hud: Label3D = get_node_or_null("../StatusHUD") as Label3D

const JellyfishVisual = preload("res://scripts/jellyfish_visual.gd")
const WhaleVisual     = preload("res://scripts/whale_visual.gd")


func _ready() -> void:
	_peer = FabricMultiplayerPeer.new()
	_peer.game_id = "abyssal_vr"
	var err := _peer.create_client(zone_host, zone_port)
	if err != OK:
		push_error("FabricClient: create_client failed: %d" % err)
		return
	# Do NOT set multiplayer.multiplayer_peer — FabricZone uses raw ENet without
	# the Godot high-level multiplayer handshake. Polling is done manually below.
	print("FabricClient: connecting to %s:%d" % [zone_host, zone_port])


func _process(_delta: float) -> void:
	if not _peer:
		return
	_peer.poll()  # drive ENet event loop (not set as multiplayer.multiplayer_peer)
	var status := _peer.get_connection_status()
	if status != MultiplayerPeer.CONNECTION_CONNECTED:
		if _frame_count % 64 == 1:
			print("FabricClient: status=%d frame=%d" % [status, _frame_count])
		_update_hud(status)
		_frame_count += 1
		return

	_frame_count += 1
	_drain_interest()
	_cull_stale_entities()
	_update_hud(status)


func _update_hud(status: int) -> void:
	if _hud == null:
		return
	var names: Array[String] = ["disconnected", "connecting", "connected"]
	var status_str: String = names[clampi(status, 0, 2)]
	_hud.text = "%s  %s:%d\nentities: %d  xing: %d/%d  snaps: %d\nframe: %d" % [
		status_str, zone_host, zone_port,
		_entity_nodes.size(), _xing_seen.size(), XING_TOTAL, _snap_count,
		_frame_count]


func _physics_process(_delta: float) -> void:
	if not _peer:
		return
	if _peer.get_connection_status() != MultiplayerPeer.CONNECTION_CONNECTED:
		return
	_send_xr_heartbeat()


# CH_PLAYER cmd=0 heartbeat: publish the local player's world-space head pose
# (sourced from the XROrigin3D's camera transform) so the zone server upserts
# a player entity row that every other zone sees via the Hilbert AOI relay.
func _send_xr_heartbeat() -> void:
	var cam: Node3D = null
	if not xr_origin.is_empty():
		cam = get_node_or_null(xr_origin)
	var pos := Vector3.ZERO
	if cam != null:
		pos = _get_branch_transform(cam).origin
	var pkt := PackedByteArray()
	pkt.resize(100)
	pkt.fill(0)
	pkt.encode_u32(0, player_id)
	pkt.encode_double(4,  pos.x)
	pkt.encode_double(12, pos.y)
	pkt.encode_double(20, pos.z)
	# vel/accel int16 at 28-39: leave 0 (no client-side extrapolation)
	# hlc at 40: leave 0 (server assigns HLC on upsert)
	# payload[0] low byte = 0 (cmd=0: player position update)
	# Match the transfer channel that spawn_stroke_knot (cmd=3) uses; the
	# server demultiplexes by cmd byte, not by transport channel.
	_peer.set_transfer_channel(2)
	_peer.set_transfer_mode(MultiplayerPeer.TRANSFER_MODE_UNRELIABLE)
	_peer.put_packet(pkt)


func _drain_interest() -> void:
	var packets: Array = _peer.drain_channel(CH_INTEREST)
	for pkt: PackedByteArray in packets:
		var offset := 0
		while offset + PACKET_ENTRY_SIZE <= pkt.size():
			var gid: int   = pkt.decode_u32(offset)
			var cx:  float = pkt.decode_double(offset + 4)
			var cy:  float = pkt.decode_double(offset + 12)
			var cz:  float = pkt.decode_double(offset + 20)
			var vx:  float = pkt.decode_s16(offset + 28) / V_SCALE
			var vy:  float = pkt.decode_s16(offset + 30) / V_SCALE
			var vz:  float = pkt.decode_s16(offset + 32) / V_SCALE
			var ax:  float = pkt.decode_s16(offset + 34) / A_SCALE
			var ay:  float = pkt.decode_s16(offset + 36) / A_SCALE
			var az:  float = pkt.decode_s16(offset + 38) / A_SCALE
			var hlc: int   = pkt.decode_u32(offset + 40)
			var p0:  int   = pkt.decode_u32(offset + PAYLOAD_OFFSET)
			var p1:  int   = pkt.decode_u32(offset + PAYLOAD_OFFSET + 4)
			var p2:  int   = pkt.decode_u32(offset + PAYLOAD_OFFSET + 8)
			offset += PACKET_ENTRY_SIZE
			_upsert_entity(gid, Vector3(cx, cy, cz), Vector3(vx, vy, vz),
					Vector3(ax, ay, az), hlc, p0, p1, p2)


var _entity_hlc: Dictionary = {}  # {int global_id -> int hlc} — stale-snapshot guard


## Returns the world-space transform of a node, working even when not yet in the
## scene tree (port of godot-vrm's get_branch_transform_for_node / 74659.patch).
## Once a node has been add_child()'d this is identical to global_transform.
func _get_branch_transform(node: Node3D) -> Transform3D:
	if node == null:
		return Transform3D.IDENTITY
	if node.is_inside_tree():
		return node.global_transform
	# Not in tree: walk up parent chain accumulating local transforms.
	var result: Transform3D = node.transform
	var current: Node3D = node
	while current.get_parent() != null:
		var parent = current.get_parent()
		if not parent is Node3D:
			break
		var parent_3d: Node3D = parent as Node3D
		if parent_3d.is_inside_tree():
			result = parent_3d.global_transform * result
			break
		result = parent_3d.transform * result
		current = parent_3d
	return result


func _upsert_entity(gid: int, pos: Vector3, vel: Vector3 = Vector3.ZERO,
		accel: Vector3 = Vector3.ZERO, hlc: int = 0,
		p0: int = 0, p1: int = 0, p2: int = 0) -> void:
	# Discard out-of-order snapshots using HLC.
	if _entity_hlc.get(gid, -1) >= hlc:
		return
	_entity_hlc[gid] = hlc
	_entity_last_seen[gid] = _frame_count

	# Phase-1 pass-condition tracking for zone-crossing entities.
	if gid >= XING_ID_LO and gid <= XING_ID_HI:
		# Snap detection: large position jump on an existing entity.
		if _xing_positions.has(gid):
			var jump: float = pos.distance_to(_xing_positions[gid])
			if jump > SNAP_THRESHOLD_M:
				_snap_count += 1
				push_warning("SNAP gid=%d jump=%.2fm frame=%d" % [gid, jump, _frame_count])
		_xing_positions[gid] = pos
		_xing_seen[gid] = true
		# Report pass condition once all 144 have been seen.
		if not _pass_logged and _xing_seen.size() >= XING_TOTAL:
			_pass_logged = true
			print("PHASE1 PASS: all %d zone-crossing entities received (snaps=%d) at frame %d" \
				% [XING_TOTAL, _snap_count, _frame_count])

	if _entity_nodes.has(gid):
		# Node is already in tree — global_position is valid.
		_entity_nodes[gid].global_position = pos
		return

	# Spawn new visual node.
	var node: Node3D
	if gid >= STROKE_ENTITY_BASE:
		# Pen stroke knot: small sphere, color from p2 (RGBA8888).
		var sphere := CSGSphere3D.new()
		sphere.radius = 0.018
		var mat := StandardMaterial3D.new()
		var r := ((p2 >> 24) & 0xFF) / 255.0
		var g := ((p2 >> 16) & 0xFF) / 255.0
		var b := ((p2 >>  8) & 0xFF) / 255.0
		var a := ( p2        & 0xFF) / 255.0
		mat.albedo_color     = Color(r, g, b, a)
		mat.transparency     = BaseMaterial3D.TRANSPARENCY_ALPHA
		mat.emission_enabled = true
		mat.emission         = Color(r, g, b) * 0.4
		sphere.material      = mat
		node = sphere
	elif gid <= 255 or (gid >= 256 and gid <= 399):
		node = JellyfishVisual.new()
		node.entity_id = gid
	else:
		node = WhaleVisual.new()
		node.entity_id = gid
		node.is_whale = (gid % 14 == 0)

	# Convert world-space pos into _root-local space using branch transform,
	# which works whether _root is in the tree or not.
	var root_xform := _get_branch_transform(_root)
	node.position = root_xform.affine_inverse() * pos
	_root.add_child(node)
	_entity_nodes[gid] = node


# Send a cmd=3 packet to the server to spawn a stroke knot.
# stroke_id: (player_id & 0xFFFF) << 16 | (counter & 0xFFFF)
# color: RGBA, encoded as RGBA8888 in payload[2].
func spawn_stroke_knot(pos: Vector3, stroke_id: int, color: Color) -> void:
	if not _peer or not _peer.get_connection_status() == MultiplayerPeer.CONNECTION_CONNECTED:
		return
	# CH_PLAYER packet: 100 bytes, same skeleton as CH_INTEREST.
	# [u32 player_id(4)][f64 cx(8)][f64 cy(8)][f64 cz(8)]
	# [i16×6 vel/accel(12)][u32 hlc(4)][u32×14 payload(56)]
	# payload[0] low byte = cmd; payload[1] = stroke_id; payload[2] = color RGBA8888
	var pkt := PackedByteArray()
	pkt.resize(100)
	pkt.fill(0)
	# player_id at 0: leave 0
	pkt.encode_double(4,  pos.x)
	pkt.encode_double(12, pos.y)
	pkt.encode_double(20, pos.z)
	# vel/accel int16 at 28–39: leave 0
	# hlc at 40: leave 0
	pkt[44] = 3  # cmd=3 in low byte of payload[0]
	pkt.encode_u32(48, stroke_id)   # payload[1]
	var rgba: int = (int(color.r * 255) << 24) | (int(color.g * 255) << 16) \
	              | (int(color.b * 255) << 8)  |  int(color.a * 255)
	pkt.encode_u32(52, rgba)        # payload[2]
	_peer.set_transfer_channel(2)
	_peer.set_transfer_mode(MultiplayerPeer.TRANSFER_MODE_UNRELIABLE)
	_peer.put_packet(pkt)


func _cull_stale_entities() -> void:
	var to_remove: Array = []
	for gid: int in _entity_last_seen:
		if _frame_count - _entity_last_seen[gid] > ENTITY_TIMEOUT_FRAMES:
			to_remove.append(gid)
	for gid: int in to_remove:
		if _entity_nodes.has(gid):
			_entity_nodes[gid].queue_free()
			_entity_nodes.erase(gid)
		_entity_last_seen.erase(gid)
		_entity_hlc.erase(gid)
