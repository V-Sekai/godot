/**************************************************************************/
/*  fabric_zone_peer_callbacks.h                                          */
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

// Function-pointer table that decouples FabricZone (in scene/main/) from
// FabricMultiplayerPeer (in modules/multiplayer_fabric/).
//
// FabricZone holds a Ref<MultiplayerPeer> and calls all zone-fabric operations
// through this struct.  The multiplayer_fabric module fills it in
// register_types.cpp using lambdas that cast the opaque MultiplayerPeer* back
// to FabricMultiplayerPeer* — safe because the module always puts a
// FabricMultiplayerPeer in that slot.

#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "scene/main/multiplayer_peer.h"

struct FabricZonePeerCallbacks {
	// Create an ENet server peer bound to p_port; returns the peer on success
	// or a null Ref on failure.
	Ref<MultiplayerPeer> (*create_server)(int p_port, int p_max_clients) = nullptr;

	// Create an ENet client peer connecting to p_address:p_port; returns the peer on success
	// or a null Ref on failure. Used by player mode — bypasses zone-mesh port arithmetic.
	Ref<MultiplayerPeer> (*create_client)(const String &p_address, int p_port) = nullptr;

	// Connect the peer to a neighbor zone server (non-blocking; retried by caller).
	// p_target_port is the actual ENet listen port of the target zone; FabricZone
	// computes it as cluster_base_port + target_zone_id so this layer never has to.
	void (*connect_to_zone)(MultiplayerPeer *p_peer, int p_target_zone_id, int p_target_port) = nullptr;

	// Returns true if the neighbor connection is established.
	bool (*is_zone_connected)(const MultiplayerPeer *p_peer, int p_zone_id) = nullptr;

	// Send raw bytes to a specific neighbor zone on p_channel.
	void (*send_to_zone_raw)(MultiplayerPeer *p_peer, int p_target_zone_id,
			int p_channel, const uint8_t *p_data, int p_size) = nullptr;

	// Broadcast raw bytes to all neighbors AND connected players on p_channel.
	void (*broadcast_raw)(MultiplayerPeer *p_peer, int p_channel,
			const uint8_t *p_data, int p_size) = nullptr;

	// Send raw bytes only to local clients (target_peer=0 on the server
	// socket), skipping neighbor zones. Used by the CH_INTEREST relay to
	// forward a packet that arrived from a neighbor without re-fanning it out.
	void (*local_broadcast_raw)(MultiplayerPeer *p_peer, int p_channel,
			const uint8_t *p_data, int p_size) = nullptr;

	// Drain all packets from p_channel and return them.
	LocalVector<Vector<uint8_t>> (*drain_channel_raw)(MultiplayerPeer *p_peer,
			int p_channel) = nullptr;

	bool is_valid() const {
		return create_server && create_client && connect_to_zone &&
				is_zone_connected && send_to_zone_raw && broadcast_raw &&
				local_broadcast_raw && drain_channel_raw;
	}
};
