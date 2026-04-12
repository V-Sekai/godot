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

} // namespace TestFabricZone

#endif // MODULE_MULTIPLAYER_FABRIC_ENABLED
