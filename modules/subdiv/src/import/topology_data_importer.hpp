/**************************************************************************/
/*  topology_data_importer.hpp                                            */
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

#ifndef TOPOLOGY_DATA_IMPORTER_H
#define TOPOLOGY_DATA_IMPORTER_H

#include "core/object/object.h"
#include "core/templates/hash_map.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh.h"

#include "../resources/topology_data_mesh.hpp"

class TopologyDataImporter : public RefCounted {
	GDCLASS(TopologyDataImporter, RefCounted);

protected:
	static void _bind_methods();

public:
	enum ImportMode {
		BAKED_SUBDIV_MESH = 0,
		IMPORTER_MESH,
	};

private:
	struct TopologySurfaceData {
		PackedVector3Array vertex_array;
		PackedVector3Array normal_array;
		PackedVector2Array uv_array;
		PackedInt32Array index_array;
		PackedInt32Array bones_array;
		Vector<float> weights_array;
	};

	struct SurfaceVertexArrays { //of imported triangle mesh
		PackedVector3Array vertex_array;
		PackedVector3Array normal_array;
		PackedInt32Array index_array;
		PackedVector2Array uv_array;
		PackedInt32Array bones_array; //could be float or int array after docs
		Vector<float> weights_array;
		SurfaceVertexArrays(const Array &p_mesh_arrays);
		SurfaceVertexArrays(){};
	};

	TopologyDataImporter::TopologySurfaceData _remove_duplicate_vertices(const SurfaceVertexArrays &surface, int32_t format);
	Array _generate_packed_blend_shapes(const Array &tri_blend_shapes,
			const PackedInt32Array &mesh_index_array, const PackedVector3Array &mesh_vertex_array);
	bool _merge_to_quads(PackedInt32Array &index_array, PackedVector2Array &uv_array, int32_t format);
	PackedInt32Array _generate_uv_index_array(PackedVector2Array &uv_array);
	int _generate_topology_surface_arrays(const SurfaceVertexArrays &surface, int32_t format, Array &surface_arrays);

	int32_t generate_fake_format(const Array &arrays) const;
	MeshInstance3D *_replace_importer_mesh_instance_with_mesh_instance(ImporterMeshInstance3D *importer_mesh_instance);

public:
	TopologyDataImporter();
	~TopologyDataImporter();

	void convert_importer_meshinstance_to_subdiv(ImporterMeshInstance3D *importer_mesh_instance, ImportMode import_mode, int32_t subdiv_level);
};

VARIANT_ENUM_CAST(TopologyDataImporter::ImportMode);

#endif
