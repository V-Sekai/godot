/**************************************************************************/
/*  fabric_zone.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "scene/main/fabric_zone_peer_callbacks.h"
#include "scene/main/scene_tree.h"

#include <thirdparty/predictive_bvh/predictive_bvh.h>

// Timing and migration constants come from predictive_bvh.h (generated from Lean):
//   PBVH_SIM_TICK_HZ, PBVH_LATENCY_TICKS, PBVH_HYSTERESIS_THRESHOLD,
//   PBVH_INTEREST_RADIUS_UM, PBVH_V_MAX_PHYSICAL, PBVH_ACCEL_FLOOR
// Spatial primitives from predictive_bvh.h: Aabb, hilbert_of_aabb,
// ghost_bound, per_entity_delta_poly (all proved in Lean).
static constexpr int INTENT_SIZE = 88; // 8 + 4 + 4 + 9*8(double) = 88 bytes

// ── real_t (meters) ↔ R128 (μm) bridge ────────────────────────────────────

static inline R128 r128_from_real_um(real_t v) {
	return r128_from_float((float)(v * 1000000.0));
}

static inline real_t r128_to_real_m(R128 v) {
	return (real_t)((int64_t)v.hi) * (real_t)0.000001;
}

// ── FabricZone ─────────────────────────────────────────────────────────────

class FabricZone : public SceneTree {
	GDCLASS(FabricZone, SceneTree);

public:
	// ── Hilbert AOI band ───────────────────────────────────────────────
	// AOI_CELLS: number of Hilbert cells of padding applied to each side
	// of a zone's own range. The relay and the neighbor-topology loop both
	// read the same band so one constant controls fanout width.
	// AOI_CELLS=1 gives two or three neighbors in a 100-zone fabric;
	// AOI_CELLS=2 on a 3-zone smoke covers the full Hilbert space (full
	// mesh by derivation, not by branch).
	static constexpr int AOI_CELLS = 2;
	// Fills [out_lo, out_hi) = Hilbert AOI band for this zone's (my_id,
	// zone_count) pair. Clamped to [0, 2^30). Static/pure so unit tests
	// can mirror the Lean theorems in Protocol/Fabric.lean without
	// constructing a SceneTree.
	static void _hilbert_aoi_band(uint32_t &out_lo, uint32_t &out_hi, int my_id, int count);

	// ── Entity (real_t, meters) ────────────────────────────────────────
	struct FabricEntity {
		real_t cx = 0.0, cy = 0.0, cz = 0.0;
		real_t vx = 0.0, vy = 0.0, vz = 0.0;
		real_t ax = 0.0, ay = 0.0, az = 0.0;
		int global_id = 0;
		// CH_INTEREST wire entry — 100 bytes:
		//   4(gid) + 3×8(cx/cy/cz f64) + 6×2(vx/vy/vz/ax/ay/az int16) + 4(hlc u32) + 14×4(payload)
		//   Offset 28: velocity/accel int16, scale V_SCALE / A_SCALE.
		//   Offset 40: hlc = tick(24b) | counter(8b) — hybrid logical clock.
		//   Offset 44: payload[14]. Unused words are zero.
		// Interpretation is entity-type specific.
		//
		// ── Generic layout (applicable to all games) ───────────────────────
		//   payload[0] = entity_class(8b) | owner_id(16b) | state_flags(8b)
		//             entity_class: game-defined type tag (0=NPC, 1=player-owned, 2=prop, 3=effect…)
		//             owner_id:     player or server that spawned this entity
		//             state_flags:  8 game-specific 1-bit flags (active, visible, interactable…)
		//   payload[1] = subtype_a(16b) | subtype_b(16b)   — game-specific secondary classification
		//   payload[2..3] = spatial_extra — game-specific packed positional/orientation data
		//   payload[4..7] = game-mode payload — health/ammo, animation state, item IDs, blendshapes…
		//
		// ── Abyssal VR Grid (CONCEPT.md) concrete layout ───────────────────
		//   jellyfish_bloom (global_id 0–255):
		//     payload[0] = class=0 | owner_id=server | bloom_phase(8b)
		//     payload[1] = kick_timer(16b) | reserved
		//   jellyfish_zone_crossing (global_id 256–399):
		//     payload[0] = class=1 | owner_id=server | crossing_flags(8b)
		//     payload[1] = group_id(4b) | waypoint_idx(6b) | reserved
		//   whale_with_sharks (global_id 400–511):
		//     payload[0] = class=2 | owner_id=server | is_whale(1b)|pod_id(7b)
		//     payload[1] = slot_idx(4b) | reserved
		//   pen stroke knots (global_id >= STROKE_ENTITY_BASE):
		//     payload[0] = class=3 | owner_id=player_id | reserved
		//     payload[1] = stroke_id (u32)
		//     payload[2] = stroke_color RGBA8888 (u32)
		//     payload[3] = anchor_cx int16 (lo) | anchor_cy int16 (hi)  ±SIM_BOUND→±32767
		//     payload[4] = anchor_cz int16 (lo) | reserved (hi)
		//
		// ── V-Sekai VR social (extension example) ──────────────────────────
		//   55 humanoid bones per player (godot-humanoid-project BoneCount).
		//   Rotation stored as 3-axis muscle triplet (bone_swing_twists Vector3 ±1),
		//   NOT as a quaternion — calculate_humanoid_rotation() reconstructs at runtime.
		//     payload[0] = class=4 | player_id(16b) | bone_index(6b)|dof_mask(2b)
		//     payload[1] = axis_x int16 (lo) | axis_y int16 (hi)   — swing ±1→±32767
		//     payload[2] = axis_z int16 (lo) | reserved (hi)        — twist axis
		//     payload[3] = held_item_left(16b) | held_item_right(16b)
		//     payload[4] = expression_preset(8b)|voice_active(1b)|group_id(7b)|anim_state(16b)
		//     payload[5..7] = blendshapes / game-mode specific
		//     payload[8..13] = reserved / future extension
		//
		uint32_t payload[14] = {};
	};

	// ── Ghost snap (real_t, meters; proved computations use R128 at call site) ─
	struct GhostSnap {
		real_t cx = 0.0, cy = 0.0, cz = 0.0;
		real_t vx = 0.0, vy = 0.0, vz = 0.0;
		real_t max_ahx = 0.0, max_ahy = 0.0, max_ahz = 0.0;
		uint32_t per_delta = 1;
		uint32_t ticks_since_snap = 0;
		real_t ghost_ey = 0.0;
		uint32_t ghost_hilbert = 0;
	};

	// ── Scenario enum ───────────────────────────────────────────────────
	enum Scenario {
		SCENARIO_DEFAULT,
		SCENARIO_JELLYFISH_BLOOM,
		SCENARIO_JELLYFISH_ZONE_CROSSING,
		SCENARIO_WHALE_WITH_SHARKS,
		SCENARIO_CURRENT_FUNNEL,
		SCENARIO_MIXED,
	};

	// ── Peer callback table — filled by multiplayer_fabric module ───────
	static FabricZonePeerCallbacks peer_callbacks;
	static void register_peer_callbacks(const FabricZonePeerCallbacks &p_cb);

private:
	// ── Constants (real_t, meters) ───────────────────────────────────────
	static constexpr real_t SIM_BOUND = 15.0;
	// Wire-encoding + physics constants derive from PBVH_*_DEFAULT (the value
	// the Lean proofs were evaluated at). These MUST be compile-time stable —
	// all peers on the fabric agree on the wire quantization scales. Runtime
	// tick-rate adaptation happens in the tick-cadence helpers below via
	// Engine::get_physics_ticks_per_second(), not in wire encoding.
	static constexpr real_t V_MAX = PBVH_V_MAX_PHYSICAL_DEFAULT * 0.000001; // m/tick
	static constexpr real_t INTEREST_RADIUS = PBVH_INTEREST_RADIUS_UM * 0.000001; // m
	static constexpr float V_SCALE = 32767.0f / (PBVH_V_MAX_PHYSICAL_DEFAULT * 1.0e-6f); // m/tick → int16
	static constexpr float A_SCALE = 32767.0f / (2.0f * PBVH_V_MAX_PHYSICAL_DEFAULT * 1.0e-6f); // m/tick² → int16
	static constexpr real_t CURRENT_FUNNEL_PEAK_V = V_MAX * 6.0; // m/tick — C7 rip-current impulse cap (60 m/s)
	static constexpr real_t ACCEL_FLOOR_M = PBVH_ACCEL_FLOOR_DEFAULT * 0.000001; // m/tick²
	static constexpr int N_TOTAL = 100000;
	static constexpr int DEFAULT_ZONE_COUNT = 3;
	// WP_PERIOD: half-cycle duration for jellyfish zone-crossing waypoint flips.
	// 10 seconds of wall time; the actual tick count is computed at use-site
	// from Engine::get_physics_ticks_per_second(). Must exceed WaypointBound.lean's
	// wpPeriodMin (travel + hysteresis + latency, all simTickHz-parametric).
	static constexpr int WP_PERIOD_SECONDS = 10;
	// Scenario animation cadences, all declared in seconds/ms and converted to ticks
	// at the use site against the engine's physics tick rate.
	static constexpr int BLOOM_BEAT_PERIOD_SECONDS = 1; // jellyfish_bloom pulse cycle
	static constexpr uint32_t BLOOM_BEAT_ON_MS = 83; // pulse-on fraction
	static constexpr int WHALE_PHASE_PERIOD_SECONDS = 2; // whale-with-sharks heading phase
	static constexpr int FUNNEL_PRONE_PERIOD_SECONDS = 3; // current_funnel prone cycle
	static constexpr int JET_BELL_PERIOD_SECONDS = 3; // jellyfish_zone_crossing bell period
	static constexpr uint32_t JET_PULSE_MS = 100; // jet thrust duration
	static constexpr uint32_t STATS_LOG_INTERVAL_SECONDS = 20; // periodic stats log cadence
	static constexpr int MAX_ZONES = 32;
	// ZONE_CAPACITY: hard upper bound on slot array size (compile-time max).
	// _zone_capacity: runtime limit set via --zone-capacity N (default = ZONE_CAPACITY).
	// Headless dedicated server: 1800 (AbyssalSLA.lean: 16 players × 56 + 904 ecosystem).
	// PCVR co-located 90 Hz: 1024 (896 player + 128 ecosystem; ~2.5ms on x86 fits 3ms budget).
	// PCVR co-located 72 Hz: 1200 (896 player + 304 ecosystem; ~3ms on x86 fits 4.7ms budget).
	// Never go below entitiesPerPlayer × targetPlayersPerZone = 896 or the SLA proof breaks.
	// Zones scale horizontally — one zone process per core per machine.
	static constexpr int ZONE_CAPACITY = 1800;
	int _zone_capacity = ZONE_CAPACITY; // set before slot allocation in initialize()
	static constexpr int STROKE_ENTITY_BASE = 1000000; // global_id >= this → pen stroke knot
	static constexpr int MAX_STROKE_KNOTS = 50; // max knots per stroke chain (snake head/tail)
	static constexpr uint32_t INTEREST_PUBLISH_INTERVAL = 1;

	// ── Zone state ───────────────────────────────────────────────────────
	int zone_id = 0;
	int zone_count = 3;
	uint16_t cluster_base_port = 17500; // base_port+zone_id = this zone's listen port
	// Centroid of each zone's entities, computed in initialize() from the
	// same N_TOTAL spawn pass. Used as waypoints for jellyfish_zone_crossing.
	real_t _zone_centroid[MAX_ZONES][3] = {};
	uint32_t tick = 0;
	uint64_t migrations = 0;
	uint64_t xing_started = 0; // OWNED→STAGING handoffs initiated this zone
	uint64_t xing_done = 0; // STAGING deactivations completed this zone
	uint64_t xing_received = 0; // inbound entities activated this zone
	bool done = false;
	bool is_player = false;
	Scenario scenario = SCENARIO_DEFAULT;

	// ── Neighbor reconnect backoff (index 0 = zone_id-1, index 1 = zone_id+1) ─
	// Starts at pbvh_latency_ticks(engine_hz), doubles on each failed attempt,
	// caps at 30 s worth of ticks. Both the initial value and the cap are
	// (re)computed at runtime against Engine::get_physics_ticks_per_second().
	static constexpr uint32_t RETRY_CAP_SECONDS = 30;
	uint32_t _retry_next[2] = { 0, 0 };
	uint32_t _retry_interval[2] = { PBVH_LATENCY_TICKS_DEFAULT, PBVH_LATENCY_TICKS_DEFAULT };

	// ── Per-neighbor STAGING latency (Resources.lean: perNeighborLatencyTicks) ──
	// Index 0 = zone_id-1, index 1 = zone_id+1.
	// Default = pbvh_latency_ticks(engine_hz) (= latencyTicksFloor, proved for 0ms RTT).
	// Updated via HLC ping/pong RTT measurement on CH_MIGRATION.
	// Formula: max(pbvh_latency_ticks(hz), ceil(rtt_ms * hz / 1000))
	// (proved: Resources.lean perNeighborLatencyTicks, per_neighbor_ge_floor)
	uint32_t _neighbor_latency_ticks[2] = { PBVH_LATENCY_TICKS_DEFAULT, PBVH_LATENCY_TICKS_DEFAULT };

	// ── HLC-based RTT ping/pong (CH_MIGRATION, 8-byte packets) ─────────────
	// Ping:    [u32 send_tick][u32 PING_MAGIC]   zone A → zone B
	// Pong:    [u32 echoed_tick][u32 PONG_MAGIC]  zone B → zone A
	// Ack:     [u32 eid][u32 ACK_MAGIC]           zone B → zone A (migration accepted)
	// Packets are < INTENT_SIZE (88 bytes) so _unpack_intent ignores them safely.
	// RTT = (current_tick - echoed_tick) ticks; latency_ticks = RTT/2 (one-way).
	static constexpr uint32_t PING_MAGIC = 0x50494E47u; // 'PING'
	static constexpr uint32_t PONG_MAGIC = 0x504F4E47u; // 'PONG'
	static constexpr uint32_t ACK_MAGIC = 0x41434B4Bu; // 'ACKK' — migration accepted by zone B
	static constexpr uint32_t PING_INTERVAL_SECONDS = 8; // wall seconds between RTT pings
	// Drain control packets (8-byte, same envelope as PING/PONG/ACK):
	//   DRAIN:     [u32 zone_id][u32 DRAIN_MAGIC]      — "I am shutting down"
	//   DRAIN_DONE:[u32 zone_id][u32 DRAIN_DONE_MAGIC]  — "I have zero entities"
	static constexpr uint32_t DRAIN_MAGIC = 0x4452414Eu; // 'DRAN'
	static constexpr uint32_t DRAIN_DONE_MAGIC = 0x44524E44u; // 'DRND'
	// Drain timeout: max wall seconds to wait for entities to flush before force-quit.
	static constexpr uint32_t DRAIN_TIMEOUT_SECONDS = 30;
	// Snapshot path: Godot Resource (.res) written by zone 0 during drain.
	static constexpr const char *SNAPSHOT_PATH = "user://fabric_snapshot.res";
	uint32_t _ping_next[2] = { 0, 0 };
	uint32_t _ping_send_tick[2] = { 0, 0 }; // tick when last ping was sent

	// ── Graceful shutdown drain state ──────────────────────────────────────
	// On SIGINT/finalize, all zones drain entities toward zone 0 via STAGING.
	// Zone 0 streams inbound entities to fabric_snapshot.bin (write-through)
	// without allocating slots. Intermediate zones forward inbound entities
	// toward zone 0 when their own slots are full (passthrough).
	bool _draining = false;
	uint32_t _drain_start_tick = 0;
	bool _neighbor_drain_done[2] = { false, false }; // index 0=zone_id-1, 1=zone_id+1
	// Zone 0 drain buffer: entities collected from all zones, saved as Resource on finalize.
	LocalVector<FabricEntity> _drain_buffer;
	// --drain-at-tick N: begin drain at tick N (0 = disabled). For testing.
	uint32_t _drain_at_tick = 0;

	// ── Server entity storage (fixed-size slot array) ───────────────────
	// Migration state machine (matches Lean: Fabric.lean MigrationState):
	//   OWNED:    active=true,  is_staging=false, is_incoming=false  (normal)
	//   STAGING:  active=true,  is_staging=true,  is_incoming=false  (zone A, sent intent)
	//   INCOMING: active=true,  is_staging=false, is_incoming=true   (zone B, received intent)
	// Transitions:
	//   OWNED → STAGING (zone A): when entity crosses Hilbert boundary after hysteresis
	//   STAGING → OWNED (zone A): on ACK from zone B, or rollback on timeout
	//   INCOMING → OWNED (zone B): one tick after receipt; sends ACK to zone A
	struct EntitySlot {
		FabricEntity entity;
		GhostSnap snap;
		uint32_t hysteresis = 0;
		bool active = false;
		// STAGING: zone A side — intent sent, waiting for ACK from zone B.
		bool is_staging = false;
		uint32_t staging_send_tick = 0;
		int migration_target_zone = -1;
		// INCOMING: zone B side — intent received, one tick before becoming OWNED.
		// On finalization (next tick), sends ACK_MAGIC back to zone A.
		bool is_incoming = false;
		int incoming_from_zone = -1;
		// Player entity slots: is_player_slot=true means position is driven by CH_PLAYER,
		// not by _step_entity. Player slots are not migrated — authority stays with connected zone.
		bool is_player_slot = false;
		uint32_t last_update_tick = 0; // for player slot idle-expiry
	};
	EntitySlot *slots = nullptr;
	int entity_count = 0;
	int free_hint = 0;

	// Stroke chain FIFOs: stroke_id → slot indices (oldest at front, newest at back).
	// push_back new head; when size > MAX_STROKE_KNOTS, deactivate front (tail).
	HashMap<uint32_t, LocalVector<int>> _stroke_chains;
	int _stroke_entity_counter = 0;

	// Player entity slot tracking: player_id → slot index.
	// Enables O(1) lookup on every CH_PLAYER update.
	// Player slots use global_id = PLAYER_ENTITY_BASE + player_id (class=1 in payload[0]).
	static constexpr int PLAYER_ENTITY_BASE = 2000000;
	// Player slot expires after 3 seconds without a CH_PLAYER update.
	static constexpr uint32_t PLAYER_SLOT_TIMEOUT_SECONDS = 3;
	HashMap<uint32_t, int> _player_slot_map; // player_id → slot index

	static constexpr int MAX_MIGRATIONS_PER_TICK = 50;
	// Migration headroom: slots reserved for inbound migrations.
	// Two components must both fit within headroom:
	//   1. Burst (jellyfish_zone_crossing): 144 entities arrive at tick ~282.
	//   2. Pre-burst drift: non-crossing entities drift via free vz into zone 1's
	//      Hilbert region. Measured: ~166 drift entities arrive before the burst.
	//      Total peak = 166 + 144 = 310. ×1.15 safety = 357 → 400.
	// Erlang-B sizing (jellyfish_bloom 256-entity burst) also satisfied: 400 > 300.
	// Effective spawn cap = _zone_capacity - MIGRATION_HEADROOM = 1800 - 400 = 1400.
	static constexpr int MIGRATION_HEADROOM = 400;

	// ── Player state (real_t, meters) ────────────────────────────────────
	int player_id = 0;
	real_t player_cx = 0.0;
	real_t player_cy = 0.0;
	real_t player_vx = 0.0;
	real_t player_vy = 0.0;
	uint64_t entities_received = 0;

	// ── Phase-1 pass condition tracking (player/observer mode) ───────────
	// Tracks jellyfish_zone_crossing entities (gid 256–399, exactly 144).
	// Indexed by gid - 256.  _p1_seen is false until the gid is first received.
	static constexpr int XING_ID_LO = 256;
	static constexpr int XING_ID_HI = 399;
	static constexpr int XING_TOTAL = 144; // XING_ID_HI - XING_ID_LO + 1
	// Snap threshold: above max 3D movement per broadcast interval.
	// sqrt(3) * V_MAX * INTEREST_PUBLISH_INTERVAL = sqrt(3)*0.15625*1 ≈ 0.27m → 0.5m margin.
	static constexpr real_t SNAP_THRESHOLD_M = 0.5;
	bool _p1_seen[XING_TOTAL] = {};
	real_t _p1_cx[XING_TOTAL] = {};
	real_t _p1_cy[XING_TOTAL] = {};
	real_t _p1_cz[XING_TOTAL] = {};
	uint32_t _p1_last_tick[XING_TOTAL] = {}; // tick of last received update per entity
	int _p1_seen_count = 0;
	int _p1_snap_count = 0;
	bool _p1_pass_logged = false;
	// Gap threshold: if entity was absent longer than this, first reappearance is
	// not a snap. Expressed in milliseconds (ENet jitter budget); converted to
	// ticks at use site via the engine's physics tick rate.
	static constexpr uint32_t SNAP_ABSENCE_MS = 200;

	// ── Networking ───────────────────────────────────────────────────────
	LocalVector<Vector<uint8_t>> intent_inbox;
	Ref<MultiplayerPeer> fabric_peer; // implementation provided via peer_callbacks

	// ── Timing ───────────────────────────────────────────────────────────
	uint64_t total_compute_us = 0;
	uint64_t total_sync_us = 0;
	uint64_t wall_start_usec = 0;

	// ── Entity stepping ──────────────────────────────────────────────────
	// p_waypoints:   zone centroids [zone][3] for migration target; null for non-crossing scenarios.
	// p_zone_count:  number of zones (chooses waypoint via shuffled cycle hash).
	// p_local_clump: centroid of tracers currently in this zone; null if none present.
	//                Used for cohesion spring — gives natural clumping without O(N²).
	static void _step_entity(FabricEntity &e, uint32_t p_tick, Scenario p_scenario, int p_n,
			const real_t p_waypoints[][3] = nullptr, int p_zone_count = 1,
			const real_t *p_local_clump = nullptr);

	// ── Ghost snap helpers ───────────────────────────────────────────────
	static uint32_t _snap_delta(real_t v, real_t ah);
	static GhostSnap _make_ghost_snap(const FabricEntity &e);
	static void _update_snap(GhostSnap &snap, const FabricEntity &e);
	static Aabb _ghost_aabb_from_snap(const GhostSnap &snap);

	static Aabb _scene_aabb();

	// ── GDScript-exposed Hilbert helpers (public for ClassDB + tests) ───
public:
	static int _zone_for_hilbert(uint32_t hcode, int count);
	static uint32_t _entity_hilbert(const FabricEntity &e);
	static ::AABB hilbert_cell_of_aabb(int p_code, int p_prefix_depth);
	static int hilbert_of_point(float p_x, float p_y, float p_z);

private:
	// ── Hilbert AOI band ────────────────────────────────────────────────
	// AOI_CELLS = number of cell_w-wide ranges of Hilbert padding added on
	// each side of a zone's own range. Relay + neighbor-topology both read
	// the same band so one constant controls the fanout width. AOI_CELLS=1
	// gives two or three neighbors in a 100-zone fabric; AOI_CELLS=2 on a
	// 3-zone smoke covers the full Hilbert space (full mesh by derivation,
	// not by branch).
	// ── Migration serialization (float wire format) ─────────────────────
	static Vector<uint8_t> _pack_intent(int eid, int to, uint32_t arrival, const FabricEntity &e);
	static bool _unpack_intent(const uint8_t *p_data, int p_size, int &r_eid, int &r_to, uint32_t &r_arrival, FabricEntity &r_entity);

	// ── Drain helpers ───────────────────────────────────────────────────
	void _begin_drain(); // enter draining mode, broadcast DRAIN_MAGIC
	void _drain_collect_entity(const FabricEntity &e); // zone 0: buffer entity for snapshot
	void _drain_save_snapshot(); // zone 0: write snapshot Resource to disk
	int _load_snapshot(); // load snapshot from disk into slots, returns entity count loaded

protected:
	static void _bind_methods();

	// ── Slot helpers for MMOG-layer subclasses ───────────────────────────
	// Finds the next free slot, reinitializes it, marks it active, and
	// increments entity_count.  Returns the slot index, or -1 when full.
	int _alloc_entity_slot();
	// Marks the slot at p_idx inactive and decrements entity_count.
	void _free_entity_slot(int p_idx);
	// Direct reference to the FabricEntity inside slot p_idx for
	// post-alloc initialization (payload, global_id, coordinates).
	FabricEntity &_slot_entity_ref(int p_idx);
	// Send raw bytes to a specific connected peer on p_channel
	// (TRANSFER_MODE_RELIABLE). Intended for targeted reliable sends such
	// as delivering the script registry to a newly-joined client.
	void _send_to_peer_raw(int p_peer_id, int p_channel, const uint8_t *p_data, int p_size);

public:
	virtual void initialize() override;
	virtual bool physics_process(double p_time) override;
	virtual void finalize() override;

	FabricZone() = default;
	~FabricZone();
};
