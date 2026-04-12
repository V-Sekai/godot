/**************************************************************************/
/*  test_fabric_zone.cpp                                                  */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_fabric_zone)

#include "modules/modules_enabled.gen.h" // For multiplayer_fabric.

#ifdef MODULE_MULTIPLAYER_FABRIC_ENABLED

#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "tests/test_utils.h"

#include "modules/multiplayer_fabric/fabric_snapshot.h"
#include "modules/multiplayer_fabric/fabric_zone.h"

namespace TestFabricZone {

// Mirror of the Lean theorems `aoiBand_covers_self` and `aoiBand_width_bound`
// in thirdparty/predictive_bvh/PredictiveBVH/Protocol/Fabric.lean. Any change
// to _hilbert_aoi_band must keep these two invariants green.

TEST_CASE("[FabricZone] Hilbert AOI band covers the zone's own Hilbert range") {
	// Single zone: band == full 30-bit Hilbert space.
	uint32_t lo = 0, hi = 0;
	FabricZone::_hilbert_aoi_band(lo, hi, 0, 1);
	CHECK(lo == 0u);
	CHECK(hi == (1u << 30));

	// 4-zone fabric: each zone's band must cover its own slice.
	for (int count : { 2, 3, 4, 8, 16, 100 }) {
		int depth = 0;
		for (uint32_t x = (uint32_t)(count - 1); x > 0; x >>= 1) {
			depth++;
		}
		const uint32_t cell_w = 1u << (30 - depth);
		for (int id = 0; id < count; ++id) {
			FabricZone::_hilbert_aoi_band(lo, hi, id, count);
			const uint32_t zone_lo = (uint32_t)id * cell_w;
			const uint32_t zone_hi = zone_lo + cell_w;
			CHECK_MESSAGE(lo <= zone_lo, "band lo must cover zone lo");
			CHECK_MESSAGE(zone_hi <= hi, "band hi must cover zone hi");
		}
	}
}

TEST_CASE("[FabricZone] Hilbert AOI band width is bounded by (1 + 2*AOI_CELLS)*cellWidth") {
	uint32_t lo = 0, hi = 0;
	for (int count : { 2, 3, 4, 8, 16, 100 }) {
		int depth = 0;
		for (uint32_t x = (uint32_t)(count - 1); x > 0; x >>= 1) {
			depth++;
		}
		const uint32_t cell_w = 1u << (30 - depth);
		const uint64_t bound = (uint64_t)(1 + 2 * FabricZone::AOI_CELLS) * (uint64_t)cell_w;
		for (int id = 0; id < count; ++id) {
			FabricZone::_hilbert_aoi_band(lo, hi, id, count);
			CHECK(hi >= lo);
			CHECK_MESSAGE((uint64_t)(hi - lo) <= bound, "band width exceeds (1+2*AOI_CELLS)*cell_w");
		}
	}
}

TEST_CASE("[FabricZone] Hilbert AOI band is clamped to [0, 2^30)") {
	uint32_t lo = 0, hi = 0;
	for (int count : { 1, 2, 3, 4, 8, 16, 100 }) {
		for (int id = 0; id < count; ++id) {
			FabricZone::_hilbert_aoi_band(lo, hi, id, count);
			CHECK(hi <= (1u << 30));
		}
	}
}

TEST_CASE("[FabricZone][Snapshot] FabricSnapshot round-trips through ResourceSaver") {
	Ref<FabricSnapshot> snap;
	snap.instantiate();

	PackedInt32Array ids;
	ids.push_back(42);
	ids.push_back(99);

	PackedFloat64Array pos;
	pos.resize(6);
	pos.set(0, 1.5);
	pos.set(1, -3.0);
	pos.set(2, 7.25); // entity 0
	pos.set(3, -10.0);
	pos.set(4, 0.5);
	pos.set(5, 14.9); // entity 1

	PackedFloat64Array vel;
	vel.resize(6);
	vel.set(0, 0.1);
	vel.set(1, -0.2);
	vel.set(2, 0.0);
	vel.set(3, 0.0);
	vel.set(4, 0.0);
	vel.set(5, 0.05);

	PackedFloat64Array acc;
	acc.resize(6);
	acc.fill(0.0);

	PackedInt32Array pay;
	pay.resize(28);
	pay.fill(0);
	pay.set(0, 0x00040001); // entity 0 payload[0]: class=4, owner_id=1

	snap->set_global_ids(ids);
	snap->set_positions(pos);
	snap->set_velocities(vel);
	snap->set_accelerations(acc);
	snap->set_payloads(pay);

	CHECK(snap->get_entity_count() == 2);

	// Save compressed.
	String path = TestUtils::get_temp_path("test_fabric_snapshot.res");
	Ref<Resource> res = snap;
	Error err = ResourceSaver::save(res, path, ResourceSaver::FLAG_COMPRESS);
	CHECK(err == OK);

	// Reload.
	Ref<FabricSnapshot> loaded = ResourceLoader::load(path);
	REQUIRE(loaded.is_valid());
	CHECK(loaded->get_version() == 1);
	CHECK(loaded->get_entity_count() == 2);

	PackedInt32Array loaded_ids = loaded->get_global_ids();
	CHECK(loaded_ids[0] == 42);
	CHECK(loaded_ids[1] == 99);

	PackedFloat64Array loaded_pos = loaded->get_positions();
	CHECK(loaded_pos[0] == doctest::Approx(1.5));
	CHECK(loaded_pos[3] == doctest::Approx(-10.0));

	PackedInt32Array loaded_pay = loaded->get_payloads();
	CHECK(loaded_pay[0] == 0x00040001);

	// Cleanup.
	Ref<DirAccess> dir = DirAccess::open(path.get_base_dir());
	if (dir.is_valid()) {
		dir->remove(path.get_file());
	}
}

TEST_CASE("[FabricZone] hilbert_of_point matches _entity_hilbert for known positions") {
	// Both paths normalize to [0,1023] and call hilbert3d, but through different
	// arithmetic (float vs R128 micrometers). They must agree for positions well
	// within the simulation bounds.
	struct Sample {
		float x, y, z;
	};
	Sample samples[] = {
		{ 0.0f, 0.0f, 0.0f }, // origin
		{ -14.0f, -14.0f, -14.0f }, // near min corner
		{ 14.0f, 14.0f, 14.0f }, // near max corner
		{ -7.0f, 0.0f, 7.0f }, // mid-space
		{ 5.0f, -3.0f, 10.0f }, // typical entity position
		{ 0.5f, 0.5f, 0.5f }, // near origin, positive quadrant
	};
	for (const Sample &s : samples) {
		int code_binding = FabricZone::hilbert_of_point(s.x, s.y, s.z);
		FabricZone::FabricEntity e;
		e.cx = (real_t)s.x;
		e.cy = (real_t)s.y;
		e.cz = (real_t)s.z;
		uint32_t code_entity = FabricZone::_entity_hilbert(e);
		CHECK_MESSAGE(code_binding == (int)code_entity,
				vformat("Mismatch at (%.1f, %.1f, %.1f): binding=%d entity=%d",
						(double)s.x, (double)s.y, (double)s.z, code_binding, (int)code_entity));
	}
}

TEST_CASE("[FabricZone] hilbert_cell_of_aabb returns AABB containing the input point") {
	// Spatial round-trip: point → hilbert code → cell AABB must contain the point.
	// This is the invariant that zone_curtain.gd relies on for label placement.
	struct Sample {
		float x, y, z;
	};
	Sample samples[] = {
		{ 0.0f, 0.0f, 0.0f },
		{ -10.0f, 5.0f, -3.0f },
		{ 12.0f, -8.0f, 1.0f },
		{ -14.5f, -14.5f, -14.5f }, // near boundary
		{ 7.5f, 7.5f, 7.5f },
	};
	int depth = 10; // full 10-bit depth
	for (const Sample &s : samples) {
		int code = FabricZone::hilbert_of_point(s.x, s.y, s.z);
		::AABB cell = FabricZone::hilbert_cell_of_aabb(code, depth);
		Vector3 pt(s.x, s.y, s.z);
		// Grow cell slightly for floating-point tolerance.
		::AABB grown = cell.grow(0.01);
		CHECK_MESSAGE(grown.has_point(pt),
				vformat("Point (%.1f,%.1f,%.1f) code=%d not in cell pos=(%.2f,%.2f,%.2f) size=(%.2f,%.2f,%.2f)",
						(double)s.x, (double)s.y, (double)s.z, code,
						cell.position.x, cell.position.y, cell.position.z,
						cell.size.x, cell.size.y, cell.size.z));
	}
}

// ── Migration state-machine tests ──────────────────────────────────────────
//
// These tests exercise the OWNED→STAGING→INCOMING→OWNED migration path
// using raw slot arrays and the public static _*_s methods. No FabricZone
// (SceneTree) objects are created — only plain data + pure functions.

// Lightweight zone state for test harness (no SceneTree dependency).
struct ZoneState {
	FabricZone::EntitySlot *slots = nullptr;
	int capacity = 1800;
	int zone_id = 0;
	int zone_count = 2;
	int entity_count = 0;
	int free_hint = 0;
	uint32_t tick = 0;
	uint64_t xing_started = 0;
	uint64_t xing_received = 0;
	uint64_t migrations = 0;
	uint32_t srtt[2] = { PBVH_LATENCY_TICKS_DEFAULT, PBVH_LATENCY_TICKS_DEFAULT };
	uint32_t rttvar[2] = { PBVH_LATENCY_TICKS_DEFAULT / 2, PBVH_LATENCY_TICKS_DEFAULT / 2 };
	bool rtt_measured[2] = { true, true }; // tests simulate measured RTT
	uint32_t hz = 60;
	LocalVector<Vector<uint8_t>> inbox;
	LocalVector<Vector<uint8_t>> outbox;

	void alloc() {
		slots = (FabricZone::EntitySlot *)memalloc(sizeof(FabricZone::EntitySlot) * capacity);
		memset(slots, 0, sizeof(FabricZone::EntitySlot) * capacity);
	}

	~ZoneState() {
		if (slots) {
			memfree(slots);
			slots = nullptr;
		}
	}
};

// Deliver outbox of src into inbox of dst (replaces ENet for tests).
static void deliver(ZoneState &src, ZoneState &dst) {
	for (uint32_t i = 0; i < src.outbox.size(); i++) {
		dst.inbox.push_back(src.outbox[i]);
	}
	src.outbox.clear();
}

TEST_CASE("[FabricZone][Migration] pack_intent round-trips through unpack_intent") {
	FabricZone::FabricEntity ent;
	ent.cx = 1.5;
	ent.cy = -3.0;
	ent.cz = 7.25;
	ent.vx = 0.1;
	ent.vy = -0.2;
	ent.vz = 0.0;
	ent.ax = 0.01;
	ent.ay = -0.005;
	ent.az = 0.003;
	ent.global_id = 42;

	Vector<uint8_t> pkt = FabricZone::_pack_intent(42, 1, 100, ent);
	CHECK(pkt.size() == INTENT_SIZE);

	int r_eid = 0, r_to = 0;
	uint32_t r_arrival = 0;
	FabricZone::FabricEntity r_ent;
	bool ok = FabricZone::_unpack_intent(pkt.ptr(), pkt.size(), r_eid, r_to, r_arrival, r_ent);
	CHECK(ok);
	CHECK(r_eid == 42);
	CHECK(r_to == 1);
	CHECK(r_arrival == 100);
	CHECK(r_ent.cx == doctest::Approx(1.5));
	CHECK(r_ent.cy == doctest::Approx(-3.0));
	CHECK(r_ent.cz == doctest::Approx(7.25));
	CHECK(r_ent.vx == doctest::Approx(0.1));
	CHECK(r_ent.vy == doctest::Approx(-0.2));
	CHECK(r_ent.vz == doctest::Approx(0.0));
	CHECK(r_ent.ax == doctest::Approx(0.01));
	CHECK(r_ent.ay == doctest::Approx(-0.005));
	CHECK(r_ent.az == doctest::Approx(0.003));
}

TEST_CASE("[FabricZone][Migration] staging timeout rolls back entity to OWNED") {
	ZoneState za;
	za.zone_id = 0;
	za.alloc();

	// Activate slot 0 and put it into STAGING state.
	za.slots[0].active = true;
	za.slots[0].entity.global_id = 1;
	za.slots[0].is_staging = true;
	za.slots[0].staging_send_tick = 0;
	za.slots[0].migration_target_zone = 1;
	za.entity_count = 1;

	// Advance tick past the timeout window (srtt=2, rttvar=1, rto=2+4*1=6).
	FabricZone::_resolve_staging_timeouts_s(za.slots, za.capacity,
			za.zone_id, za.srtt, za.rttvar, za.rtt_measured, za.hz, 100);

	CHECK_MESSAGE(!za.slots[0].is_staging, "Entity should roll back to OWNED after timeout");
	CHECK_MESSAGE(za.slots[0].hysteresis == 0, "Hysteresis should reset on rollback");
	CHECK_MESSAGE(za.slots[0].migration_target_zone == -1, "Target zone should clear on rollback");
}

TEST_CASE("[FabricZone][Migration] 144 entities all land within timeout window") {
	ZoneState za, zb;
	za.zone_id = 0;
	za.zone_count = 2;
	za.alloc();

	zb.zone_id = 1;
	zb.zone_count = 2;
	zb.alloc();

	// Populate Zone A with 1400 entities; last 144 are crossing-ready.
	for (int i = 0; i < 1400; i++) {
		za.slots[i].active = true;
		za.slots[i].entity.global_id = i;
		za.entity_count++;
	}
	for (int i = 1400 - 144; i < 1400; i++) {
		za.slots[i].hysteresis = 1000; // well past threshold
		za.slots[i].entity.cx = 14.0; // position in Zone B's Hilbert range
	}

	// Populate Zone B with 1400 entities.
	for (int i = 0; i < 1400; i++) {
		zb.slots[i].active = true;
		zb.slots[i].entity.global_id = 10000 + i;
		zb.entity_count++;
	}

	int total_received = 0;
	for (int t = 0; t < 32 && total_received < 144; t++) {
		// Zone A: collect intents.
		FabricZone::_collect_migration_intents_s(za.slots, za.capacity,
				za.zone_id, za.zone_count, za.srtt,
				za.tick, 60, 50, za.xing_started, za.migrations, za.outbox);

		// Deliver outbox → inbox.
		deliver(za, zb);

		// Zone B: accept incoming intents.
		int accepted = FabricZone::_accept_incoming_intents_s(zb.slots, zb.capacity,
				zb.entity_count, zb.free_hint, zb.zone_id,
				zb.inbox, zb.xing_received);
		total_received += accepted;

		// Zone A: resolve staging timeouts.
		FabricZone::_resolve_staging_timeouts_s(za.slots, za.capacity,
				za.zone_id, za.srtt, za.rttvar, za.rtt_measured, za.hz, za.tick);

		za.tick++;
		zb.tick++;
	}

	CHECK_MESSAGE(total_received == 144, vformat("Expected 144 migrations but got %d", total_received));
}

TEST_CASE("[FabricZone][Migration] Zone B at capacity rejects intent gracefully") {
	ZoneState za, zb;
	za.zone_id = 0;
	za.alloc();

	zb.zone_id = 1;
	zb.alloc();

	// Zone A: 10 entities.
	for (int i = 0; i < 10; i++) {
		za.slots[i].active = true;
		za.slots[i].entity.global_id = i;
		za.entity_count++;
	}

	// Zone B: completely full.
	for (int i = 0; i < 1800; i++) {
		zb.slots[i].active = true;
		zb.slots[i].entity.global_id = 10000 + i;
		zb.entity_count++;
	}

	// Manually stage slot 9 on Zone A.
	za.slots[9].is_staging = true;
	za.slots[9].staging_send_tick = 0;
	za.slots[9].migration_target_zone = 1;

	// Fabricate intent and push into Zone B's inbox.
	Vector<uint8_t> pkt = FabricZone::_pack_intent(za.slots[9].entity.global_id, 1, 5, za.slots[9].entity);
	zb.inbox.push_back(pkt);

	// Zone B tries to accept — should reject (no free slots).
	int accepted = FabricZone::_accept_incoming_intents_s(zb.slots, zb.capacity,
			zb.entity_count, zb.free_hint, zb.zone_id,
			zb.inbox, zb.xing_received);
	CHECK_MESSAGE(accepted == 0, "Zone B at capacity should reject all intents");
	CHECK_MESSAGE(zb.entity_count == 1800, "Zone B entity count should not change");

	// Zone A rolls back via timeout.
	FabricZone::_resolve_staging_timeouts_s(za.slots, za.capacity,
			za.zone_id, za.srtt, za.rttvar, za.rtt_measured, za.hz, 100);
	CHECK_MESSAGE(!za.slots[9].is_staging, "Zone A should roll back after Zone B rejection");
}

TEST_CASE("[FabricZone][Migration] outbound budget queues excess entities across ticks") {
	ZoneState za;
	za.zone_id = 0;
	za.zone_count = 2;
	za.alloc();

	// 120 entities, all crossing-ready.
	for (int i = 0; i < 120; i++) {
		za.slots[i].active = true;
		za.slots[i].entity.global_id = i;
		za.slots[i].hysteresis = 1000;
		za.slots[i].entity.cx = 14.0; // in Zone B's Hilbert range
		za.entity_count++;
	}

	int total_sent = 0;

	// Tick 1: should send exactly 50 (budget).
	int sent_t1 = FabricZone::_collect_migration_intents_s(za.slots, za.capacity,
			za.zone_id, za.zone_count, za.srtt,
			0, 60, 50, za.xing_started, za.migrations, za.outbox);
	CHECK_MESSAGE(sent_t1 == 50, vformat("Tick 1: expected 50 intents, got %d", sent_t1));
	total_sent += sent_t1;

	// Tick 2: another 50.
	int sent_t2 = FabricZone::_collect_migration_intents_s(za.slots, za.capacity,
			za.zone_id, za.zone_count, za.srtt,
			1, 60, 50, za.xing_started, za.migrations, za.outbox);
	CHECK_MESSAGE(sent_t2 == 50, vformat("Tick 2: expected 50 intents, got %d", sent_t2));
	total_sent += sent_t2;

	// Tick 3: remaining 20.
	int sent_t3 = FabricZone::_collect_migration_intents_s(za.slots, za.capacity,
			za.zone_id, za.zone_count, za.srtt,
			2, 60, 50, za.xing_started, za.migrations, za.outbox);
	CHECK_MESSAGE(sent_t3 == 20, vformat("Tick 3: expected 20 intents, got %d", sent_t3));
	total_sent += sent_t3;

	CHECK_MESSAGE(total_sent == 120, vformat("Total sent should be 120, got %d", total_sent));
}

} // namespace TestFabricZone

#endif // MODULE_MULTIPLAYER_FABRIC_ENABLED
