/**************************************************************************/
/*  fabric_mmog_peer.cpp                                                  */
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

#include "fabric_mmog_peer.h"

#include "core/math/math_funcs.h"
#include "core/object/class_db.h"

void FabricMMOGPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("encode_muscle_triplet_component", "value"),
			&FabricMMOGPeer::encode_muscle_triplet_component);
	ClassDB::bind_method(D_METHOD("decode_muscle_triplet_component", "encoded"),
			&FabricMMOGPeer::decode_muscle_triplet_component);
	ClassDB::bind_method(D_METHOD("encode_humanoid_bone", "player_id", "bone_index", "swing_twist", "held_left", "held_right"),
			&FabricMMOGPeer::encode_humanoid_bone);

	BIND_CONSTANT(WIRE_PACKET_BYTES);
	BIND_CONSTANT(WIRE_OFFSET_ID);
	BIND_CONSTANT(WIRE_OFFSET_CX);
	BIND_CONSTANT(WIRE_OFFSET_CY);
	BIND_CONSTANT(WIRE_OFFSET_CZ);
	BIND_CONSTANT(WIRE_OFFSET_VX);
	BIND_CONSTANT(WIRE_OFFSET_VY);
	BIND_CONSTANT(WIRE_OFFSET_VZ);
	BIND_CONSTANT(WIRE_OFFSET_AX);
	BIND_CONSTANT(WIRE_OFFSET_AY);
	BIND_CONSTANT(WIRE_OFFSET_AZ);
	BIND_CONSTANT(WIRE_OFFSET_HLC);
	BIND_CONSTANT(WIRE_OFFSET_PAYLOAD);

	BIND_CONSTANT(HLC_TICK_BITS);
	BIND_CONSTANT(HLC_COUNTER_BITS);

	BIND_CONSTANT(MUSCLE_TRIPLET_COMPONENT_BITS);
	BIND_CONSTANT(MUSCLE_TRIPLET_SCALE);

	BIND_ENUM_CONSTANT(CMD_POSITION_ONLY);
	BIND_ENUM_CONSTANT(CMD_CURRENT_FUNNEL);
	BIND_ENUM_CONSTANT(CMD_NUDGE);
	BIND_ENUM_CONSTANT(CMD_SPAWN_STROKE);
}

int FabricMMOGPeer::encode_muscle_triplet_component(float p_value) const {
	const float clamped = CLAMP(p_value, -1.0f, 1.0f);
	return (int)Math::round(clamped * (float)MUSCLE_TRIPLET_SCALE);
}

float FabricMMOGPeer::decode_muscle_triplet_component(int p_encoded) const {
	return (float)p_encoded / (float)MUSCLE_TRIPLET_SCALE;
}

Vector<uint8_t> FabricMMOGPeer::encode_humanoid_bone(int p_player_id, int p_bone_index,
		const Vector3 &p_swing_twist, int p_held_left, int p_held_right) const {
	// STUB: produce a WIRE_PACKET_BYTES-length buffer with payload[0..4]
	// populated per the humanoid bone layout. For now return a zero-filled
	// packet of the correct size so callers can exercise the interface.
	Vector<uint8_t> packet;
	packet.resize(WIRE_PACKET_BYTES);
	memset(packet.ptrw(), 0, WIRE_PACKET_BYTES);
	return packet;
}
