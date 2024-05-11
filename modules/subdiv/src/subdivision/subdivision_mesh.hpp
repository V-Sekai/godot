/**************************************************************************/
/*  subdivision_mesh.hpp                                                  */
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

#ifndef SUBDIVISION_MESH_HPP
#define SUBDIVISION_MESH_HPP

#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh.h"

#include "core/templates/hash_map.h"

#include "../resources/topology_data_mesh.hpp"

//SubdivisionMesh is only for subdividing ImporterQuadMeshes
class SubdivisionMesh : public RefCounted {
	GDCLASS(SubdivisionMesh, RefCounted);
	RID source_mesh; //ImporterQuadMesh
	RID subdiv_mesh; //generated triangle mesh

	int current_level = -1;

protected:
	static void _bind_methods();
	Array _get_subdivided_arrays(const Array &p_arrays, int p_level, int32_t p_format, bool calculate_normals, int topology_type);

	Vector<int64_t> subdiv_vertex_count; //variables used for compatibility with mesh
	Vector<int64_t> subdiv_index_count;

public:
	SubdivisionMesh();
	~SubdivisionMesh();

	RID get_rid() const;
	void set_rid(RID p_rid);
	void update_subdivision(Ref<TopologyDataMesh> p_mesh, int p_level);
	void _update_subdivision(Ref<TopologyDataMesh> p_mesh, int p_level, const Vector<Array> &cached_data_arrays);
	void update_subdivision_vertices(int p_surface, const PackedVector3Array &new_vertex_array,
			const PackedInt32Array &index_array, int topology_type);
	void clear();

	int64_t surface_get_vertex_array_size(int p_surface) const;
	int64_t surface_get_index_array_size(int p_surface) const;
};

#endif // SUBDIVISION_MESH_HPP
