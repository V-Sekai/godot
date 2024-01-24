/**************************************************************************/
/*  vox_importer.h                                                        */
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

#ifndef VOX_IMPORTER_H
#define VOX_IMPORTER_H

#include "editor/import/3d/resource_importer_scene.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/importer_mesh.h"
#include <editor/import/editor_import_plugin.h>

#include "streams/vox_data.h"

class VoxelBuffer;
class VoxelMesher;
class VoxelVoxImporter : public EditorSceneFormatImporter {
	GDCLASS(VoxelVoxImporter, EditorSceneFormatImporter);

	struct VoxMesh {
		Ref<ImporterMesh> mesh;
		Vector3 pivot;
	};

	static Error process_scene_node_recursively(const vox::Data &data, int node_id,
			Node3D *parent_node,
			Node3D *&root_node, int depth,
			const Vector<VoxMesh> &meshes);
	static void add_mesh_instance(Ref<ImporterMesh> mesh, Node *parent, Node *owner,
			Vector3 offset);
	static Ref<ImporterMesh>
	build_mesh(VoxelBuffer &voxels, VoxelMesher &mesher,
			Ref<Image> &out_atlas);

public:
	virtual uint32_t get_import_flags() const override {
		return EditorSceneFormatImporter::IMPORT_SCENE;
	}
	virtual void get_extensions(List<String> *r_extensions) const override {
		ERR_FAIL_NULL(r_extensions);
		r_extensions->push_back("vox");
	}
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, const HashMap<StringName, Variant> &p_options, List<String> *r_missing_deps, Error *r_err) override;
	VoxelVoxImporter() {}
};

#endif // VOX_IMPORTER_H
