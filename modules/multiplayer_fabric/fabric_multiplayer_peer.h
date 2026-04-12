/**************************************************************************/
/*  fabric_multiplayer_peer.h                                             */
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
#include "scene/main/multiplayer_peer.h"

// Logical channels via set_transfer_channel() + set_transfer_mode().
// Godot's ENetMultiplayerPeer reserves ENet wire channels 0-1 for system messages.
// set_transfer_channel(N) → ENet wire channel N+1. Use N≥1 for safe user channels.
static constexpr int CH_MIGRATION = 1; // → ENet wire ch 2, reliable STAGING intents
static constexpr int CH_INTEREST = 2; // → ENet wire ch 3, unreliable entity snapshots
static constexpr int CH_PLAYER = 3; // → ENet wire ch 4, unreliable player state

/// Zone-fabric multiplayer peer. Wraps an inner MultiplayerPeer (ENet or
/// WebRTC) and adds zone-to-zone neighbor connections with channel-sorted
/// inboxes. One server + one client per neighbor, all on the same port.
///
/// GDScript usage:
///   var peer = FabricMultiplayerPeer.new()
///   peer.create_server(port)
///   peer.connect_to_zone(neighbor_zone_id)
///   peer.poll()
class FabricMultiplayerPeer : public MultiplayerPeer {
	GDCLASS(FabricMultiplayerPeer, MultiplayerPeer);

	String game_id;

	// Inner peer (ENetMultiplayerPeer by default; swap to WebRTC).
	Ref<MultiplayerPeer> server_peer;

	struct NeighborConn {
		Ref<MultiplayerPeer> peer;
		bool connected = false;
	};
	HashMap<int, NeighborConn> neighbors;

	// Channel-sorted inboxes filled during poll(). One per logical channel so
	// drain_channel_raw(CH_PLAYER) never returns CH_INTEREST packets and vice
	// versa — otherwise 100-byte entity snapshots get parsed as player upserts
	// and phantom-fill the zone to capacity.
	LocalVector<Vector<uint8_t>> migration_inbox; // CH_MIGRATION (reliable)
	LocalVector<Vector<uint8_t>> interest_inbox; // CH_INTEREST  (unreliable)
	LocalVector<Vector<uint8_t>> player_inbox; // CH_PLAYER    (unreliable)

	// Current packet state for MultiplayerPeer interface.
	Vector<uint8_t> current_packet_data;
	int32_t current_packet_peer = 0;
	int32_t current_packet_channel = 0;
	TransferMode current_packet_mode = TRANSFER_MODE_RELIABLE;

	uint16_t base_port = 0;

	void _poll_peer(Ref<MultiplayerPeer> p_peer);

protected:
	static void _bind_methods();

public:
	Error create_server(int p_port, int p_max_clients = 32);
	Error create_client(const String &p_address, int p_port);

	void set_game_id(const String &p_id);
	String get_game_id() const;

	// Zone-fabric API (GDScript-callable).
	void connect_to_zone(int p_target_zone_id);
	// Internal: connect to a neighbor by explicit port (avoids base_port arithmetic).
	void connect_to_zone_at(int p_target_zone_id, int p_target_port);
	bool is_zone_connected(int p_zone_id) const;
	void send_to_zone(int p_target_zone_id, int p_channel, const PackedByteArray &p_data);
	void broadcast_to_zones(int p_channel, const PackedByteArray &p_data);
	Array drain_channel(int p_channel);

	// Raw-pointer overloads for internal C++ callers (avoids PackedByteArray alloc).
	void send_to_zone_raw(int p_target_zone_id, int p_channel, const uint8_t *p_data, int p_size);
	void broadcast_raw(int p_channel, const uint8_t *p_data, int p_size);
	// Local-only fanout: writes once to the server peer with target_peer=0 so
	// attached clients receive the packet, but does NOT forward to neighbor
	// zones. Used by the CH_INTEREST relay to duplicate a neighbor row to local
	// clients without re-fanning it back across the link it arrived on.
	void local_broadcast_raw(int p_channel, const uint8_t *p_data, int p_size);
	LocalVector<Vector<uint8_t>> drain_channel_raw(int p_channel);

	// MultiplayerPeer interface.
	virtual void set_target_peer(int p_peer_id) override;
	virtual int get_packet_peer() const override;
	virtual TransferMode get_packet_mode() const override;
	virtual int get_packet_channel() const override;
	virtual void disconnect_peer(int p_peer, bool p_force = false) override;
	virtual bool is_server() const override;
	virtual void poll() override;
	virtual void close() override;
	virtual int get_unique_id() const override;
	virtual ConnectionStatus get_connection_status() const override;
	virtual bool is_server_relay_supported() const override;

	// PacketPeer interface.
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) override;
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) override;
	virtual int get_available_packet_count() const override;
	virtual int get_max_packet_size() const override;

	FabricMultiplayerPeer() = default;
	~FabricMultiplayerPeer() = default;
};
