/**************************************************************************/
/*  fabric_mmog_zone.h                                                    */
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

#include "modules/multiplayer_fabric/fabric_zone.h"
#include "modules/multiplayer_fabric_mmog/fabric_mmog_asset.h"

// FabricMMOGZone — MMOG-layer zone server.
//
// Specialization of FabricZone that adds the V-Sekai load target:
//   • 55 skeletal bones + 1 root = 56 networked entities per player
//   • script_slot/script_version routing for godot-sandbox ELF delivery
//   • reliable script registry broadcast at session start (CH_MIGRATION)
//   • asset manifest handshake wired to FabricMMOGAsset
//
// All quantitative numbers below are the canonical values for the MMOG layer.
// CONCEPT_MMOG.md describes the "why"; this header is the "what".
class FabricMMOGZone : public FabricZone {
	GDCLASS(FabricMMOGZone, FabricZone);

public:
	// ── V-Sekai load target ─────────────────────────────────────────────
	// 55 humanoid bones from godot-humanoid-project's human_trait.gd BoneCount.
	static constexpr int HUMANOID_BONE_COUNT = 55;
	// Root entity + 55 bones → one networked entity per bone plus one anchor.
	static constexpr int ENTITIES_PER_PLAYER = HUMANOID_BONE_COUNT + 1; // 56
	// Load test goal: 100 concurrent players per zone.
	static constexpr int TARGET_PLAYERS_PER_ZONE = 100;
	// Experimentally determined slot budget: 100 × 56 = 5600 bones; add
	// ecosystem headroom (NPCs, props) → 1800 is the provisioning number
	// that holds on M2 at 3.35 ms per step and fits the 50 ms @ 20 Hz budget.
	static constexpr int ZONE_CAPACITY_TARGET = 1800;

	// ── Entity class tag for humanoid bones ─────────────────────────────
	// Lives in payload[0]'s entity_class(8b) field. See CH_INTEREST layout.
	static constexpr uint8_t ENTITY_CLASS_HUMANOID_BONE = 4;
	// global_id base for bone entities: player_id * ENTITIES_PER_PLAYER + bone.
	// Chosen well above FabricZone::STROKE_ENTITY_BASE (1M) and
	// PLAYER_ENTITY_BASE (2M) to avoid collisions.
	static constexpr int HUMANOID_BONE_ENTITY_BASE = 3000000;

	// ── Script registry entry (one per godot-sandbox ELF or scene asset) ──
	// Wire format per CONCEPT_MMOG.md §Asset delivery:
	//   [slot: u16][index_chunk_id: 32B SHA-512/256][uro_uuid: 16B] = 50 B
	struct ScriptRegistryEntry {
		uint16_t slot = 0;
		uint8_t index_chunk_id[FabricMMOGAsset::REGISTRY_INDEX_ID_BYTES] = {};
		uint8_t uro_uuid[FabricMMOGAsset::REGISTRY_URO_UUID_BYTES] = {};
	};
	static_assert(sizeof(ScriptRegistryEntry) ==
					FabricMMOGAsset::REGISTRY_SLOT_BYTES +
							FabricMMOGAsset::REGISTRY_INDEX_ID_BYTES +
							FabricMMOGAsset::REGISTRY_URO_UUID_BYTES,
			"ScriptRegistryEntry size mismatch");

	FabricMMOGZone() {}
	~FabricMMOGZone() {}

	// ── Host-side API ────────────────────────────────────────────────────
	// Register a script/asset slot prior to any client joining.
	// p_index_chunk_id must be exactly REGISTRY_INDEX_ID_BYTES (32) bytes.
	// p_uro_uuid must be exactly REGISTRY_URO_UUID_BYTES (16) bytes.
	void register_script(int p_slot, const PackedByteArray &p_index_chunk_id,
			const PackedByteArray &p_uro_uuid);

	// ── Host-side stubs ─────────────────────────────────────────────────
	// Allocate ENTITIES_PER_PLAYER bone-entity slots for a joining player.
	// Called once per player connection; returns the first global_id so the
	// client can index the range [base, base + ENTITIES_PER_PLAYER).
	int spawn_humanoid_entities_for_player(int p_player_id);

	// Free bone-entity slots when a player disconnects.
	void despawn_humanoid_entities_for_player(int p_player_id);

	// Broadcast the script/scene registry to a newly-joined peer on
	// CH_MIGRATION (reliable). Registry = ordered list of
	// (slot, index_chunk_id[32], uro_uuid[16]) = 50 bytes per entry.
	void send_script_registry(int p_peer_id);

private:
	// player_id → list of bone-entity slot indices (length = ENTITIES_PER_PLAYER).
	// Populated by spawn_humanoid_entities_for_player; cleared by despawn.
	HashMap<int, LocalVector<int>> _player_bone_slots;
	// Ordered list of registered script/asset entries broadcast on CH_MIGRATION
	// to each newly-joined peer via send_script_registry().
	LocalVector<ScriptRegistryEntry> _script_registry;

protected:
	static void _bind_methods();
};
