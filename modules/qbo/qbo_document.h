/**************************************************************************/
/*  qbo_document.h                                                        */
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

#include "modules/gltf/gltf_document.h"

class QBODocument : public GLTFDocument {
	GDCLASS(QBODocument, GLTFDocument);

	static Error _parse_motion(Ref<FileAccess> f, List<Skeleton3D *> &r_skeletons, AnimationPlayer **r_animation);
	static Error _parse_material_library(const String &p_path, HashMap<String, Ref<StandardMaterial3D>> &material_map, List<String> *r_missing_deps);
	static Error _parse_obj(Ref<FileAccess> f, const String &p_base_path, List<Ref<ImporterMesh>> &r_meshes, bool p_single_mesh, bool p_generate_tangents, bool p_optimize, Vector3 p_scale_mesh, Vector3 p_offset_mesh, bool p_disable_compression, List<String> *r_missing_deps, List<Skeleton3D *> &r_skeletons, AnimationPlayer **r_animation);

public:
	Error parse_qbo_data(Ref<FileAccess> f, Ref<GLTFState> p_state, uint32_t p_flags, String p_base_path, String p_path);
	virtual Error append_from_file(const String& p_path, Ref<GLTFState> p_state, uint32_t p_flags = 0, const String& p_base_path = String()) override;
	virtual Error append_from_buffer(const PackedByteArray &p_bytes, const String& p_base_path, Ref<GLTFState> p_state, uint32_t p_flags = 0) override;
	QBODocument() {}
	virtual ~QBODocument();

protected:
	Node *root = nullptr;
};
