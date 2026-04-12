/**************************************************************************/
/*  fabric_mmog_peer.h                                                    */
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

#include "core/object/ref_counted.h"
#include "core/variant/binder_common.h"

// FabricMMOGPeer — client-side MMOG wire helper.
//
// Wraps the transport-layer FabricMultiplayerPeer and knows how to encode /
// decode the 100-byte packet into MMOG-layer concepts: humanoid bone
// payloads, muscle triplet rotations, script slot references, and command
// routing. This class is the canonical source for all numeric wire facts.
// CONCEPT_MMOG.md stays qualitative; the numbers live here.
class FabricMMOGPeer : public RefCounted {
	GDCLASS(FabricMMOGPeer, RefCounted);

public:
	// ── Wire packet skeleton (CH_PLAYER and CH_INTEREST) ────────────────
	static constexpr int WIRE_PACKET_BYTES = 100;

	// Field offsets inside the 100-byte packet:
	static constexpr int WIRE_OFFSET_ID = 0; // 4 B   global_id or player_id
	static constexpr int WIRE_OFFSET_CX = 4; // 8 B   f64 meters
	static constexpr int WIRE_OFFSET_CY = 12; // 8 B
	static constexpr int WIRE_OFFSET_CZ = 20; // 8 B
	static constexpr int WIRE_OFFSET_VX = 28; // 2 B   int16, V_SCALE
	static constexpr int WIRE_OFFSET_VY = 30; // 2 B
	static constexpr int WIRE_OFFSET_VZ = 32; // 2 B
	static constexpr int WIRE_OFFSET_AX = 34; // 2 B   int16, A_SCALE
	static constexpr int WIRE_OFFSET_AY = 36; // 2 B
	static constexpr int WIRE_OFFSET_AZ = 38; // 2 B
	static constexpr int WIRE_OFFSET_HLC = 40; // 4 B   u32
	static constexpr int WIRE_OFFSET_PAYLOAD = 44; // 56 B  14 × u32

	// ── HLC layout ──────────────────────────────────────────────────────
	static constexpr int HLC_TICK_BITS = 24;
	static constexpr int HLC_COUNTER_BITS = 8;
	static constexpr uint32_t HLC_TICK_MASK = 0xFFFFFF00u;
	static constexpr uint32_t HLC_COUNTER_MASK = 0x000000FFu;

	// ── Command set (low byte of payload[0] at WIRE_OFFSET_PAYLOAD + 0) ─
	enum Command {
		CMD_POSITION_ONLY = 0,
		CMD_CURRENT_FUNNEL = 1, // C7 rip-current velocity spike, interest-radius scoped
		CMD_NUDGE = 2, // payload[1] = target_entity_id (u32)
		CMD_SPAWN_STROKE = 3, // payload[1] = stroke_id (u32), payload[2] = RGBA8888 (u32)
	};

	// ── Muscle triplet (swing-twist rotation encoding) ──────────────────
	// Three int16 values, each representing a component in [-1, 1]:
	//   axis_x, axis_y  — swing (transverse to reference axis)
	//   axis_z          — twist (sin(θ/2) around reference axis)
	// Implicit scale: int16(v × 32767), decoded as i / 32767.0.
	static constexpr int MUSCLE_TRIPLET_COMPONENT_BITS = 16;
	static constexpr int MUSCLE_TRIPLET_SCALE = 32767;

	// ── Encoding / decoding stubs ───────────────────────────────────────
	int encode_muscle_triplet_component(float p_value) const;
	float decode_muscle_triplet_component(int p_encoded) const;

	// Encode a humanoid bone entity payload into the 56-byte payload region.
	// Returns a byte buffer of length WIRE_PACKET_BYTES ready to send on
	// CH_PLAYER (client→zone) or consume on CH_INTEREST (zone→peer).
	Vector<uint8_t> encode_humanoid_bone(int p_player_id, int p_bone_index,
			const Vector3 &p_swing_twist,
			int p_held_left, int p_held_right) const;

	FabricMMOGPeer() {}
	~FabricMMOGPeer() {}

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(FabricMMOGPeer::Command);
