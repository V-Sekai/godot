/**************************************************************************/
/*  merge.h                                                               */
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

#ifndef MERGE_H
#define MERGE_H
/*
xatlas
https://github.com/jpcy/xatlas
Copyright (c) 2018 Jonathan Young
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/*
thekla_atlas
https://github.com/Thekla/thekla_atlas
MIT License
Copyright (c) 2013 Thekla, Inc
Copyright NVIDIA Corporation 2006 -- Ignacio Castano <icastano@nvidia.com>
*/

#include "core/object/ref_counted.h"

#include "core/math/vector2.h"
#include "core/object/ref_counted.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/main/node.h"

#include "thirdparty/xatlas/xatlas.h"
#include <cstdint>

class MeshTextureAtlas {
public:
	struct TextureData {
		uint16_t width;
		uint16_t height;
		int num_components;
		Ref<Image> image;
	};
	struct ModelVertex {
		Vector3 pos;
		Vector3 normal;
		Vector2 uv;
	};
	struct MeshState {
		Ref<Mesh> mesh;
		NodePath path;
		int32_t index_offset = 0;
		MeshInstance3D *mesh_instance;
		bool operator==(const MeshState &rhs) const;
		bool is_valid() const;
	};
	struct MaterialImageCache {
		Ref<Image> albedo_img;
	};
	struct MeshMerge {
		Vector<MeshState> meshes;
		int vertex_count = 0;
	};
	struct MeshMergeState {
		Vector<MeshMerge> mesh_items;
		Node *root = nullptr;
	};
	static constexpr float TEXEL_SIZE = 5.0f;

	struct AtlasLookupTexel {
		uint16_t material_index = 0;
		uint16_t x = 0;
		uint16_t y = 0;
	};
	struct AtlasTextureArguments {
		Ref<Image> atlas_data;
		Ref<Image> source_texture;
		AtlasLookupTexel *atlas_lookup = nullptr;
		uint16_t material_index = 0;
		Vector2 source_uvs[3];
		uint32_t atlas_width = 0;
		uint32_t atlas_height = 0;
	};

	struct MergeState {
		Node *p_root = nullptr;
		xatlas::Atlas *atlas = nullptr;
		Vector<MeshState> &r_mesh_items;
		Array &vertex_to_material;
		const Vector<Vector<Vector2> > uvs;
		const Vector<Vector<ModelVertex> > &model_vertices;
		String p_name;
		const xatlas::PackOptions &pack_options;
		Vector<AtlasLookupTexel> &atlas_lookup;
		Vector<Ref<Material> > &material_cache;
		HashMap<String, Ref<Image> > texture_atlas;
		HashMap<int32_t, MaterialImageCache> material_image_cache;
	};
	static bool set_atlas_texel(void *param, int x, int y, const Vector3 &bar, const Vector3 &dx, const Vector3 &dy, float coverage);
	static Pair<int, int> calculate_coordinates(const Vector2 &sourceUv, int width, int height);
	MeshTextureAtlas();
	static Node *merge_meshes(Node *p_root);

private:
	static int godot_xatlas_print(const char *p_print_string, ...);
	static Vector2 interpolate_source_uvs(const Vector3 &bar, const AtlasTextureArguments *args);
	static Ref<Image> dilate_image(Ref<Image> source_image);
	static void _find_all_mesh_instances(Vector<MeshMerge> &r_items, Node *p_current_node, const Node *p_owner);
	static void _generate_texture_atlas(MergeState &state, String texture_type);
	static Ref<Image> _get_source_texture(MergeState &state, Ref<BaseMaterial3D> material);
	static Error _generate_atlas(const int32_t p_num_meshes, Vector<Vector<Vector2> > &r_uvs, xatlas::Atlas *atlas, const Vector<MeshState> &r_meshes, const Vector<Ref<Material> > material_cache,
			xatlas::PackOptions &pack_options);
	static void write_uvs(const Vector<MeshState> &p_mesh_items, Vector<Vector<Vector2> > &uv_groups, Array &r_vertex_to_material, Vector<Vector<ModelVertex> > &r_model_vertices);
	static void map_mesh_to_index_to_material(const Vector<MeshState> &mesh_items, Array &vertex_to_material, Vector<Ref<Material> > &material_cache);
	static Node *_output_mesh_atlas(MergeState &state, int p_count);

protected:
	static void _bind_methods();
};

#endif // MERGE_H
