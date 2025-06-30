/**************************************************************************/
/*  voxel_terrain_multiplayer_synchronizer.h                              */
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

#include "../../storage/voxel_data_block.h"
#include "../../util/containers/std_unordered_map.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/node.h"
#include "../../util/math/box3i.h"

#ifdef TOOLS_ENABLED
#include "../../util/godot/core/version.h"
#endif

namespace zylann::voxel {

class VoxelTerrain;

// Implements multiplayer replication for `VoxelTerrain`
class VoxelTerrainMultiplayerSynchronizer : public Node {
	GDCLASS(VoxelTerrainMultiplayerSynchronizer, Node)
public:
	VoxelTerrainMultiplayerSynchronizer();

	bool is_server() const;

	void send_block(int viewer_peer_id, const VoxelDataBlock &data_block, Vector3i bpos);
	void send_area(Box3i voxel_box);

#ifdef TOOLS_ENABLED
#if defined(ZN_GODOT)
	PackedStringArray get_configuration_warnings() const override;
#elif defined(ZN_GODOT_EXTENSION)
	PackedStringArray _get_configuration_warnings() const override;
#endif
	void get_configuration_warnings(PackedStringArray &warnings) const;
#endif

private:
	void _notification(int p_what);

	void process();

	void _b_receive_blocks(PackedByteArray message_data);
	void _b_receive_area(PackedByteArray message_data);

	static void _bind_methods();

	VoxelTerrain *_terrain = nullptr;
	int _rpc_channel = 0;

	struct DeferredBlockMessage {
		PackedByteArray data;
	};

	StdUnorderedMap<int, StdVector<DeferredBlockMessage>> _deferred_block_messages_per_peer;
};

} // namespace zylann::voxel
