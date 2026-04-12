/**************************************************************************/
/*  fabric_multiplayer_peer.cpp                                           */
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

#include "fabric_multiplayer_peer.h"

#include "core/object/class_db.h"
#include "core/string/print_string.h"

#include "modules/enet/enet_multiplayer_peer.h"

#include <cstring>

void FabricMultiplayerPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_server", "port", "max_clients"), &FabricMultiplayerPeer::create_server, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("create_client", "address", "port"), &FabricMultiplayerPeer::create_client);
	ClassDB::bind_method(D_METHOD("set_game_id", "id"), &FabricMultiplayerPeer::set_game_id);
	ClassDB::bind_method(D_METHOD("get_game_id"), &FabricMultiplayerPeer::get_game_id);
	ClassDB::bind_method(D_METHOD("connect_to_zone", "target_zone_id"), &FabricMultiplayerPeer::connect_to_zone);
	ClassDB::bind_method(D_METHOD("send_to_zone", "target_zone_id", "channel", "data"), &FabricMultiplayerPeer::send_to_zone);
	ClassDB::bind_method(D_METHOD("broadcast_to_zones", "channel", "data"), &FabricMultiplayerPeer::broadcast_to_zones);
	ClassDB::bind_method(D_METHOD("drain_channel", "channel"), &FabricMultiplayerPeer::drain_channel);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "game_id"), "set_game_id", "get_game_id");
}

// ---------------------------------------------------------------------------
// create_server / create_client — wraps ENetMultiplayerPeer.
// ---------------------------------------------------------------------------

Error FabricMultiplayerPeer::create_server(int p_port, int p_max_clients) {
	Ref<ENetMultiplayerPeer> enet;
	enet.instantiate();
	Error err = enet->create_server(p_port, p_max_clients);
	if (err != OK) {
		return err;
	}
	server_peer = enet;
	base_port = (uint16_t)p_port;
	return OK;
}

Error FabricMultiplayerPeer::create_client(const String &p_address, int p_port) {
	Ref<ENetMultiplayerPeer> enet;
	enet.instantiate();
	Error err = enet->create_client(p_address, p_port);
	if (err != OK) {
		return err;
	}
	server_peer = enet;
	base_port = (uint16_t)p_port;
	return OK;
}

void FabricMultiplayerPeer::set_game_id(const String &p_id) {
	game_id = p_id;
}

String FabricMultiplayerPeer::get_game_id() const {
	return game_id;
}

// ---------------------------------------------------------------------------
// Zone-fabric API.
// ---------------------------------------------------------------------------

void FabricMultiplayerPeer::connect_to_zone(int p_target_zone_id) {
	connect_to_zone_at(p_target_zone_id, (int)base_port + p_target_zone_id);
}

void FabricMultiplayerPeer::connect_to_zone_at(int p_target_zone_id, int p_target_port) {
	HashMap<int, NeighborConn>::Iterator it = neighbors.find(p_target_zone_id);
	if (it != neighbors.end()) {
		if (it->value.connected) {
			return; // Already connected.
		}
		// If ENet is still mid-handshake, don't interrupt it.
		if (it->value.peer.is_valid() &&
				it->value.peer->get_connection_status() != CONNECTION_DISCONNECTED) {
			return;
		}
		// Definitively disconnected — remove stale entry so we can retry.
		neighbors.remove(it);
	}
	Ref<ENetMultiplayerPeer> enet;
	enet.instantiate();
	Error err = enet->create_client("127.0.0.1", p_target_port);
	if (err != OK) {
		return; // Neighbor not up yet — caller will retry next period.
	}
	NeighborConn conn;
	conn.peer = enet;
	neighbors.insert(p_target_zone_id, conn);
}

bool FabricMultiplayerPeer::is_zone_connected(int p_zone_id) const {
	HashMap<int, NeighborConn>::ConstIterator it = neighbors.find(p_zone_id);
	return it != neighbors.end() && it->value.connected;
}

void FabricMultiplayerPeer::send_to_zone(int p_target_zone_id, int p_channel, const PackedByteArray &p_data) {
	HashMap<int, NeighborConn>::Iterator it = neighbors.find(p_target_zone_id);
	if (it == neighbors.end() || !it->value.connected) {
		return;
	}
	Ref<MultiplayerPeer> peer = it->value.peer;
	peer->set_transfer_channel(p_channel);
	peer->set_transfer_mode(p_channel == CH_MIGRATION ? TRANSFER_MODE_RELIABLE : TRANSFER_MODE_UNRELIABLE);
	peer->set_target_peer(1);
	peer->put_packet(p_data.ptr(), p_data.size());
}

void FabricMultiplayerPeer::broadcast_to_zones(int p_channel, const PackedByteArray &p_data) {
	for (KeyValue<int, NeighborConn> &kv : neighbors) {
		if (kv.value.connected) {
			send_to_zone(kv.key, p_channel, p_data);
		}
	}
	// Also broadcast to all inbound peers on server.
	if (server_peer.is_valid()) {
		server_peer->set_transfer_channel(p_channel);
		server_peer->set_transfer_mode(p_channel == CH_MIGRATION ? TRANSFER_MODE_RELIABLE : TRANSFER_MODE_UNRELIABLE);
		server_peer->set_target_peer(0);
		server_peer->put_packet(p_data.ptr(), p_data.size());
	}
}

Array FabricMultiplayerPeer::drain_channel(int p_channel) {
	Array result;
	LocalVector<Vector<uint8_t>> *inbox_ptr = nullptr;
	switch (p_channel) {
		case CH_MIGRATION:
			inbox_ptr = &migration_inbox;
			break;
		case CH_INTEREST:
			inbox_ptr = &interest_inbox;
			break;
		case CH_PLAYER:
			inbox_ptr = &player_inbox;
			break;
		default:
			return result;
	}
	LocalVector<Vector<uint8_t>> &inbox = *inbox_ptr;
	for (unsigned int i = 0; i < (unsigned int)inbox.size(); i++) {
		PackedByteArray pba;
		pba.resize(inbox[i].size());
		if (inbox[i].size() > 0) {
			memcpy(pba.ptrw(), inbox[i].ptr(), inbox[i].size());
		}
		result.push_back(pba);
	}
	inbox.clear();
	return result;
}

// ---------------------------------------------------------------------------
// Raw-pointer overloads for internal C++ callers.
// ---------------------------------------------------------------------------

void FabricMultiplayerPeer::send_to_zone_raw(int p_target_zone_id, int p_channel, const uint8_t *p_data, int p_size) {
	HashMap<int, NeighborConn>::Iterator it = neighbors.find(p_target_zone_id);
	if (it == neighbors.end() || !it->value.connected) {
		return;
	}
	Ref<MultiplayerPeer> peer = it->value.peer;
	peer->set_transfer_channel(p_channel);
	peer->set_transfer_mode(p_channel == CH_MIGRATION ? TRANSFER_MODE_RELIABLE : TRANSFER_MODE_UNRELIABLE);
	peer->set_target_peer(1);
	peer->put_packet(p_data, p_size);
}

void FabricMultiplayerPeer::broadcast_raw(int p_channel, const uint8_t *p_data, int p_size) {
	for (KeyValue<int, NeighborConn> &kv : neighbors) {
		if (kv.value.connected) {
			send_to_zone_raw(kv.key, p_channel, p_data, p_size);
		}
	}
	local_broadcast_raw(p_channel, p_data, p_size);
}

void FabricMultiplayerPeer::local_broadcast_raw(int p_channel, const uint8_t *p_data, int p_size) {
	if (server_peer.is_valid()) {
		server_peer->set_transfer_channel(p_channel);
		server_peer->set_transfer_mode(p_channel == CH_MIGRATION ? TRANSFER_MODE_RELIABLE : TRANSFER_MODE_UNRELIABLE);
		server_peer->set_target_peer(0);
		server_peer->put_packet(p_data, p_size);
	}
}

LocalVector<Vector<uint8_t>> FabricMultiplayerPeer::drain_channel_raw(int p_channel) {
	LocalVector<Vector<uint8_t>> result;
	LocalVector<Vector<uint8_t>> *inbox_ptr = nullptr;
	switch (p_channel) {
		case CH_MIGRATION:
			inbox_ptr = &migration_inbox;
			break;
		case CH_INTEREST:
			inbox_ptr = &interest_inbox;
			break;
		case CH_PLAYER:
			inbox_ptr = &player_inbox;
			break;
		default:
			return result;
	}
	LocalVector<Vector<uint8_t>> &inbox = *inbox_ptr;
	for (uint32_t i = 0; i < inbox.size(); i++) {
		result.push_back(inbox[i]);
	}
	inbox.clear();
	return result;
}

// ---------------------------------------------------------------------------
// _poll_peer — drain packets from a peer into channel-sorted inboxes.
// ---------------------------------------------------------------------------

void FabricMultiplayerPeer::_poll_peer(Ref<MultiplayerPeer> p_peer) {
	if (p_peer.is_null() || p_peer->get_connection_status() == MultiplayerPeer::CONNECTION_DISCONNECTED) {
		return;
	}
	p_peer->poll();

	for (int i = 0; i < 100; i++) {
		if (p_peer->get_available_packet_count() <= 0) {
			break;
		}
		int ch = p_peer->get_packet_channel();
		const uint8_t *buf = nullptr;
		int size = 0;
		Error err = p_peer->get_packet(&buf, size);
		if (err != OK || size <= 0) {
			break;
		}
		Vector<uint8_t> pkt;
		pkt.resize(size);
		memcpy(pkt.ptrw(), buf, size);

		switch (ch) {
			case CH_MIGRATION:
				migration_inbox.push_back(pkt);
				break;
			case CH_INTEREST:
				interest_inbox.push_back(pkt);
				break;
			case CH_PLAYER:
				player_inbox.push_back(pkt);
				break;
			default:
				break; // Unknown channel — drop.
		}
	}
}

// ---------------------------------------------------------------------------
// MultiplayerPeer interface.
// ---------------------------------------------------------------------------

void FabricMultiplayerPeer::poll() {
	_poll_peer(server_peer);
	for (KeyValue<int, NeighborConn> &kv : neighbors) {
		NeighborConn &conn = kv.value;
		if (conn.peer.is_null() || conn.peer->get_connection_status() == MultiplayerPeer::CONNECTION_DISCONNECTED) {
			continue;
		}
		conn.peer->poll();
		if (!conn.connected &&
				conn.peer->get_connection_status() == CONNECTION_CONNECTED) {
			conn.connected = true;
			print_line(vformat("[FabricMultiplayerPeer] connected to zone %d", kv.key));
		}
		_poll_peer(conn.peer);
	}
}

void FabricMultiplayerPeer::set_target_peer(int p_peer_id) {
	if (server_peer.is_valid()) {
		server_peer->set_target_peer(p_peer_id);
	}
}

int FabricMultiplayerPeer::get_packet_peer() const {
	return current_packet_peer;
}

MultiplayerPeer::TransferMode FabricMultiplayerPeer::get_packet_mode() const {
	return current_packet_mode;
}

int FabricMultiplayerPeer::get_packet_channel() const {
	return current_packet_channel;
}

void FabricMultiplayerPeer::disconnect_peer(int p_peer, bool p_force) {
	if (server_peer.is_valid()) {
		server_peer->disconnect_peer(p_peer, p_force);
	}
}

bool FabricMultiplayerPeer::is_server() const {
	return server_peer.is_valid() && server_peer->is_server();
}

void FabricMultiplayerPeer::close() {
	if (server_peer.is_valid()) {
		server_peer->close();
		server_peer.unref();
	}
	for (KeyValue<int, NeighborConn> &kv : neighbors) {
		if (kv.value.peer.is_valid()) {
			kv.value.peer->close();
		}
	}
	neighbors.clear();
	migration_inbox.clear();
	interest_inbox.clear();
	player_inbox.clear();
}

int FabricMultiplayerPeer::get_unique_id() const {
	return server_peer.is_valid() ? server_peer->get_unique_id() : 0;
}

MultiplayerPeer::ConnectionStatus FabricMultiplayerPeer::get_connection_status() const {
	return server_peer.is_valid() ? server_peer->get_connection_status() : CONNECTION_DISCONNECTED;
}

bool FabricMultiplayerPeer::is_server_relay_supported() const {
	return false;
}

Error FabricMultiplayerPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	// Drain in priority order: migration (reliable) → interest → player.
	if (migration_inbox.size() > 0) {
		current_packet_data = migration_inbox[0];
		migration_inbox.remove_at(0);
		current_packet_channel = CH_MIGRATION;
		current_packet_mode = TRANSFER_MODE_RELIABLE;
	} else if (interest_inbox.size() > 0) {
		current_packet_data = interest_inbox[0];
		interest_inbox.remove_at(0);
		current_packet_channel = CH_INTEREST;
		current_packet_mode = TRANSFER_MODE_UNRELIABLE;
	} else if (player_inbox.size() > 0) {
		current_packet_data = player_inbox[0];
		player_inbox.remove_at(0);
		current_packet_channel = CH_PLAYER;
		current_packet_mode = TRANSFER_MODE_UNRELIABLE;
	} else {
		*r_buffer = nullptr;
		r_buffer_size = 0;
		return ERR_UNAVAILABLE;
	}
	*r_buffer = current_packet_data.ptr();
	r_buffer_size = current_packet_data.size();
	return OK;
}

Error FabricMultiplayerPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(server_peer.is_null(), ERR_UNCONFIGURED);
	return server_peer->put_packet(p_buffer, p_buffer_size);
}

int FabricMultiplayerPeer::get_available_packet_count() const {
	return migration_inbox.size() + interest_inbox.size() + player_inbox.size();
}

int FabricMultiplayerPeer::get_max_packet_size() const {
	return 1 << 24; // 16 MB.
}
