/**************************************************************************/
/*  register_types.cpp                                                    */
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

// Copyright (c) 2026-present K. S. Ernest (iFire) Lee
// SPDX-License-Identifier: MIT

#include "register_types.h"

#include "fabric_multiplayer_peer.h"
#include "fabric_snapshot.h"
#include "fabric_zone.h"

#include "core/object/class_db.h"

void initialize_multiplayer_fabric_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	GDREGISTER_CLASS(FabricMultiplayerPeer);
	GDREGISTER_CLASS(FabricSnapshot);
	GDREGISTER_CLASS(FabricZone);

	// Wire FabricZone's peer callback table to FabricMultiplayerPeer. All
	// zone-fabric operations go through this struct so FabricZone never
	// depends on the ENet-backed FabricMultiplayerPeer directly.
	FabricZonePeerCallbacks cb;

	cb.create_server = [](int p_port, int p_max_clients) -> Ref<MultiplayerPeer> {
		Ref<FabricMultiplayerPeer> peer;
		peer.instantiate();
		if (peer->create_server(p_port, p_max_clients) != OK) {
			return Ref<MultiplayerPeer>();
		}
		return peer;
	};

	cb.create_client = [](const String &p_address, int p_port) -> Ref<MultiplayerPeer> {
		Ref<FabricMultiplayerPeer> peer;
		peer.instantiate();
		if (peer->create_client(p_address, p_port) != OK) {
			return Ref<MultiplayerPeer>();
		}
		return peer;
	};

	cb.connect_to_zone = [](MultiplayerPeer *p_peer, int p_target_zone_id, int p_target_port) {
		static_cast<FabricMultiplayerPeer *>(p_peer)->connect_to_zone_at(p_target_zone_id, p_target_port);
	};

	cb.is_zone_connected = [](const MultiplayerPeer *p_peer, int p_zone_id) -> bool {
		return static_cast<const FabricMultiplayerPeer *>(p_peer)->is_zone_connected(p_zone_id);
	};

	cb.send_to_zone_raw = [](MultiplayerPeer *p_peer, int p_target_zone_id,
								  int p_channel, const uint8_t *p_data, int p_size) {
		static_cast<FabricMultiplayerPeer *>(p_peer)->send_to_zone_raw(p_target_zone_id, p_channel, p_data, p_size);
	};

	cb.broadcast_raw = [](MultiplayerPeer *p_peer, int p_channel,
							   const uint8_t *p_data, int p_size) {
		static_cast<FabricMultiplayerPeer *>(p_peer)->broadcast_raw(p_channel, p_data, p_size);
	};

	cb.local_broadcast_raw = [](MultiplayerPeer *p_peer, int p_channel,
									 const uint8_t *p_data, int p_size) {
		static_cast<FabricMultiplayerPeer *>(p_peer)->local_broadcast_raw(p_channel, p_data, p_size);
	};

	cb.drain_channel_raw = [](MultiplayerPeer *p_peer, int p_channel) -> LocalVector<Vector<uint8_t>> {
		return static_cast<FabricMultiplayerPeer *>(p_peer)->drain_channel_raw(p_channel);
	};

	FabricZone::register_peer_callbacks(cb);
}

void uninitialize_multiplayer_fabric_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
