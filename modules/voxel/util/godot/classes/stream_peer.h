/**************************************************************************/
/*  stream_peer.h                                                         */
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

#if defined(ZN_GODOT)
#include <core/io/stream_peer.h>
#elif defined(ZN_GODOT_EXTENSION)
#include "../core/packed_arrays.h"
#include <godot_cpp/classes/stream_peer.hpp>
using namespace godot;
#endif

#include "../../containers/span.h"

namespace zylann::godot {

inline void stream_peer_put_data(StreamPeer &peer, Span<const uint8_t> src) {
#if defined(ZN_GODOT)
	peer.put_data(src.data(), src.size());
#elif defined(ZN_GODOT_EXTENSION)
	PackedByteArray bytes;
	copy_to(bytes, src);
	peer.put_data(bytes);
#endif
}

inline Error stream_peer_get_data(StreamPeer &peer, Span<uint8_t> dst) {
#if defined(ZN_GODOT)
	return peer.get_data(dst.data(), dst.size());
#elif defined(ZN_GODOT_EXTENSION)
	PackedByteArray bytes = peer.get_data(dst.size());
	copy_to(dst, bytes);
	if (int64_t(dst.size()) != bytes.size()) {
		// That's what Godot returns in core
		return ERR_INVALID_PARAMETER;
	}
	return OK;
#endif
}

} // namespace zylann::godot
