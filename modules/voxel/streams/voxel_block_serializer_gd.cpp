/**************************************************************************/
/*  voxel_block_serializer_gd.cpp                                         */
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

#include "voxel_block_serializer_gd.h"
#include "../util/godot/classes/stream_peer.h"
#include "../util/godot/core/packed_arrays.h"
#include "voxel_block_serializer.h"

using namespace zylann::godot;

namespace zylann::voxel::godot {

int VoxelBlockSerializer::serialize_to_stream_peer(Ref<StreamPeer> peer, Ref<VoxelBuffer> voxel_buffer, bool compress) {
	ERR_FAIL_COND_V(voxel_buffer.is_null(), 0);
	ERR_FAIL_COND_V(peer.is_null(), 0);

	if (compress) {
		BlockSerializer::SerializeResult res = BlockSerializer::serialize_and_compress(voxel_buffer->get_buffer());
		ERR_FAIL_COND_V(!res.success, -1);
		stream_peer_put_data(**peer, to_span(res.data));
		return res.data.size();

	} else {
		BlockSerializer::SerializeResult res = BlockSerializer::serialize(voxel_buffer->get_buffer());
		ERR_FAIL_COND_V(!res.success, -1);
		stream_peer_put_data(**peer, to_span(res.data));
		return res.data.size();
	}
}

void VoxelBlockSerializer::deserialize_from_stream_peer(
		Ref<StreamPeer> peer,
		Ref<VoxelBuffer> voxel_buffer,
		int size,
		bool decompress
) {
	ERR_FAIL_COND(voxel_buffer.is_null());
	ERR_FAIL_COND(peer.is_null());
	ERR_FAIL_COND(size <= 0);

	if (decompress) {
		StdVector<uint8_t> &compressed_data = BlockSerializer::get_tls_compressed_data();
		compressed_data.resize(size);
		const Error err = stream_peer_get_data(**peer, to_span(compressed_data));
		ERR_FAIL_COND(err != OK);
		const bool success =
				BlockSerializer::decompress_and_deserialize(to_span(compressed_data), voxel_buffer->get_buffer());
		ERR_FAIL_COND(!success);

	} else {
		StdVector<uint8_t> &data = BlockSerializer::get_tls_data();
		data.resize(size);
		const Error err = stream_peer_get_data(**peer, to_span(data));
		ERR_FAIL_COND(err != OK);
		BlockSerializer::deserialize(to_span(data), voxel_buffer->get_buffer());
	}
}

PackedByteArray VoxelBlockSerializer::serialize_to_byte_array(Ref<VoxelBuffer> voxel_buffer, bool compress) {
	ERR_FAIL_COND_V(voxel_buffer.is_null(), PackedByteArray());

	PackedByteArray bytes;
	if (compress) {
		BlockSerializer::SerializeResult res = BlockSerializer::serialize_and_compress(voxel_buffer->get_buffer());
		ERR_FAIL_COND_V(!res.success, PackedByteArray());
		copy_to(bytes, to_span(res.data));

	} else {
		BlockSerializer::SerializeResult res = BlockSerializer::serialize(voxel_buffer->get_buffer());
		ERR_FAIL_COND_V(!res.success, PackedByteArray());
		copy_to(bytes, to_span(res.data));
	}
	return bytes;
}

void VoxelBlockSerializer::deserialize_from_byte_array(
		PackedByteArray bytes,
		Ref<VoxelBuffer> voxel_buffer,
		bool decompress
) {
	ERR_FAIL_COND(voxel_buffer.is_null());
	ERR_FAIL_COND(bytes.size() == 0);

	Span<const uint8_t> bytes_span = Span<const uint8_t>(bytes.ptr(), bytes.size());

	if (decompress) {
		const bool success = BlockSerializer::decompress_and_deserialize(bytes_span, voxel_buffer->get_buffer());
		ERR_FAIL_COND(!success);

	} else {
		BlockSerializer::deserialize(bytes_span, voxel_buffer->get_buffer());
	}
}

void VoxelBlockSerializer::_bind_methods() {
	auto cname = VoxelBlockSerializer::get_class_static();

	// Reasons for using methods with StreamPeer:
	// - Convenience, if you do write to a peer already
	// - Avoiding an allocation. When serializing to a PackedByteArray, the Godot API incurs allocating that
	// temporary array every time.
	ClassDB::bind_static_method(
			cname,
			D_METHOD("serialize_to_stream_peer", "peer", "voxel_buffer", "compress"),
			&VoxelBlockSerializer::serialize_to_stream_peer
	);
	ClassDB::bind_static_method(
			cname,
			D_METHOD("deserialize_from_stream_peer", "peer", "voxel_buffer", "size", "decompress"),
			&VoxelBlockSerializer::deserialize_from_stream_peer
	);

	ClassDB::bind_static_method(
			cname,
			D_METHOD("serialize_to_byte_array", "voxel_buffer", "compress"),
			&VoxelBlockSerializer::serialize_to_byte_array
	);
	ClassDB::bind_static_method(
			cname,
			D_METHOD("deserialize_from_byte_array", "bytes", "voxel_buffer", "decompress"),
			&VoxelBlockSerializer::deserialize_from_byte_array
	);
}

} // namespace zylann::voxel::godot
