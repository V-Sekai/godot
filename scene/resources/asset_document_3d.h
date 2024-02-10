/**************************************************************************/
/*  asset_document_3d.h                                                   */
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

#ifndef ASSET_DOCUMENT_3D_H
#define ASSET_DOCUMENT_3D_H

#include "scene/main/node.h"
#include "scene/resources/asset_state_3d.h"

class AssetDocument3D : public Resource {
	GDCLASS(AssetDocument3D, Resource);

protected:
	static void _bind_methods();

public:
	virtual Error append_data_from_file(String p_path, Ref<AssetState3D> p_state, uint32_t p_flags = 0, String p_base_path = String()) = 0;
	virtual Error append_data_from_buffer(PackedByteArray p_bytes, String p_base_path, Ref<AssetState3D> p_state, uint32_t p_flags = 0) = 0;
	virtual Error append_data_from_scene(Node *p_node, Ref<AssetState3D> p_state, uint32_t p_flags = 0) = 0;

public:
	virtual Node *create_scene(Ref<AssetState3D> p_state, float p_bake_fps = 30.0f, bool p_trimming = false, bool p_remove_immutable_tracks = true) = 0;
	virtual PackedByteArray create_buffer(Ref<AssetState3D> p_state) = 0;
	virtual Error write_asset_to_filesystem(Ref<AssetState3D> p_state, const String &p_path) = 0;
};

#endif // ASSET_DOCUMENT_3D_H
