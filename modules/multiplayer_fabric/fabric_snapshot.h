/**************************************************************************/
/*  fabric_snapshot.h                                                     */
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

#include "core/io/resource.h"

// FabricSnapshot: Godot Resource holding a complete fabric entity state dump.
// Written by zone 0 during graceful shutdown (drain consolidation).
// Loaded by zone 0 on restart to re-distribute entities across zones.
// Saves as .res (binary, zstd-compressed via ResourceSaver::FLAG_COMPRESS).
//
// Entity data is stored in parallel packed arrays for efficient serialization:
//   global_ids[i]         — entity global ID
//   positions[i*3..i*3+2] — cx, cy, cz (meters, f64)
//   velocities[i*3..i*3+2] — vx, vy, vz (meters/tick, f64)
//   accelerations[i*3..i*3+2] — ax, ay, az (meters/tick², f64)
//   payloads[i*14..i*14+13] — 14 u32 payload words

class FabricSnapshot : public Resource {
	GDCLASS(FabricSnapshot, Resource);

public:
	// Version 1: initial format (global_ids + positions + velocities + accelerations + payloads).
	// Bump when changing the layout so loaders can reject incompatible snapshots.
	static constexpr int CURRENT_VERSION = 1;

private:
	int version = CURRENT_VERSION;

	PackedInt32Array global_ids;
	PackedFloat64Array positions; // interleaved cx,cy,cz per entity
	PackedFloat64Array velocities; // interleaved vx,vy,vz per entity
	PackedFloat64Array accelerations; // interleaved ax,ay,az per entity
	PackedInt32Array payloads; // 14 ints per entity, interleaved

protected:
	static void _bind_methods();

public:
	void set_version(int p_version) { version = p_version; }
	int get_version() const { return version; }

	void set_global_ids(const PackedInt32Array &p_ids) { global_ids = p_ids; }
	PackedInt32Array get_global_ids() const { return global_ids; }

	void set_positions(const PackedFloat64Array &p_pos) { positions = p_pos; }
	PackedFloat64Array get_positions() const { return positions; }

	void set_velocities(const PackedFloat64Array &p_vel) { velocities = p_vel; }
	PackedFloat64Array get_velocities() const { return velocities; }

	void set_accelerations(const PackedFloat64Array &p_acc) { accelerations = p_acc; }
	PackedFloat64Array get_accelerations() const { return accelerations; }

	void set_payloads(const PackedInt32Array &p_pay) { payloads = p_pay; }
	PackedInt32Array get_payloads() const { return payloads; }

	int get_entity_count() const { return global_ids.size(); }
};
