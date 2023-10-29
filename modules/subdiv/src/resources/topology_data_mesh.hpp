/**************************************************************************/
/*  topology_data_mesh.hpp                                                */
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

#ifndef TOPOLOGY_DATA_MESH_H
#define TOPOLOGY_DATA_MESH_H

#include "core/io/resource.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"

class TopologyDataMesh : public Resource {
	GDCLASS(TopologyDataMesh, Resource);

public:
	//extra space for uv index array
	enum ArrayType {
		ARRAY_VERTEX = Mesh::ARRAY_VERTEX,
		ARRAY_NORMAL = Mesh::ARRAY_NORMAL,
		ARRAY_TANGENT = Mesh::ARRAY_TANGENT,
		ARRAY_COLOR = Mesh::ARRAY_COLOR,
		ARRAY_TEX_UV = Mesh::ARRAY_TEX_UV,
		ARRAY_TEX_UV2 = Mesh::ARRAY_TEX_UV2,
		ARRAY_CUSTOM0 = Mesh::ARRAY_CUSTOM0,
		ARRAY_CUSTOM1 = Mesh::ARRAY_CUSTOM1,
		ARRAY_CUSTOM2 = Mesh::ARRAY_CUSTOM2,
		ARRAY_CUSTOM3 = Mesh::ARRAY_CUSTOM3,
		ARRAY_BONES = Mesh::ARRAY_BONES,
		ARRAY_WEIGHTS = Mesh::ARRAY_WEIGHTS,
		ARRAY_INDEX = Mesh::ARRAY_INDEX,
		ARRAY_UV_INDEX = Mesh::ARRAY_MAX, //just an index array for uv's (ARRAY_INDEX does not work with uv's anymore in favour of face connection)
		ARRAY_MAX = Mesh::ARRAY_MAX + 1
	};

	enum {
		TOPOLOGY_DATA_MESH_TRIANGLE = 0,
		TOPOLOGY_DATA_MESH_QUAD = 1
	};
	struct Surface {
		Array arrays;
		Array blend_shape_data; //Array[Array][PackedVector3Array]
		Ref<Material> material;
		String name;
		int32_t flags = 0;
		AABB aabb;
		int32_t format;
		int topology_type;
	};

protected:
	Vector<Surface> surfaces;
	Array blend_shapes; //is Vector<StringName>, but that caused casting issues

	void _set_data(const Dictionary &p_data);
	Dictionary _get_data() const;
	static void _bind_methods();

public:
	struct TopologySurfaceData {
		PackedVector3Array vertex_array;
		PackedVector3Array normal_array;
		PackedVector2Array uv_array;
		PackedInt32Array uv_index_array;
		PackedInt32Array index_array;
		PackedInt32Array bones_array;
		Vector<float> weights_array;
		TopologySurfaceData(Array p_mesh_arrays) {
			vertex_array = p_mesh_arrays[TopologyDataMesh::ARRAY_VERTEX];
			normal_array = p_mesh_arrays[TopologyDataMesh::ARRAY_NORMAL];
			index_array = p_mesh_arrays[TopologyDataMesh::ARRAY_INDEX];
			uv_array = p_mesh_arrays[TopologyDataMesh::ARRAY_TEX_UV];
			uv_index_array = p_mesh_arrays[TopologyDataMesh::ARRAY_UV_INDEX];
			if (p_mesh_arrays[TopologyDataMesh::ARRAY_BONES])
				bones_array = p_mesh_arrays[TopologyDataMesh::ARRAY_BONES];

			if (p_mesh_arrays[TopologyDataMesh::ARRAY_WEIGHTS])
				weights_array = p_mesh_arrays[TopologyDataMesh::ARRAY_WEIGHTS];
		}
	};

	Array surface_get_arrays(int p_surface) const;

	void add_surface(const Array &p_arrays, const Array &p_blend_shapes,
			const Ref<Material> &p_material, const String &p_name, int32_t p_format, int32_t p_topology_type);
	String surface_get_name(int p_surface) const;
	void surface_set_name(int p_surface, const String &p_name);
	void surface_set_current_vertex_array(int p_surface, const PackedVector3Array &p_vertex_array);
	PackedVector3Array surface_get_current_vertex_array(int p_surface, const PackedVector3Array &p_vertex_array);
	int surface_get_length(int p_surface);
	int64_t get_surface_count() const;
	int64_t surface_get_format(int64_t index) const;
	void surface_set_material(int64_t index, const Ref<Material> &material);
	Ref<Material> surface_get_material(int64_t index) const;
	void surface_set_topology_type(int64_t index, int32_t p_topology_type);
	int32_t surface_get_topology_type(int64_t index) const;

	int64_t get_blend_shape_count() const;
	Array surface_get_blend_shape_arrays(int64_t surface_index) const;
	Array surface_get_single_blend_shape_array(int64_t surface_index, int64_t blend_shape_idx) const;
	StringName get_blend_shape_name(int64_t index) const;
	void set_blend_shape_name(int64_t index, const StringName &p_name);
	//only use after adding blend shape to surface
	void add_blend_shape_name(const StringName &p_name);
	void clear();

	TopologyDataMesh();
	~TopologyDataMesh();
};

VARIANT_ENUM_CAST(TopologyDataMesh::ArrayType);
#endif
