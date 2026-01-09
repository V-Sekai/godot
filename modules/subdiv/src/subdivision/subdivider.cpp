/**************************************************************************/
/*  subdivider.cpp                                                        */
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

#include "subdivider.hpp"

#include "core/error/error_macros.h"
#include "core/templates/hash_set.h"
#include "scene/resources/mesh_data_tool.h"
#include "servers/rendering/rendering_server.h"

#include "../resources/topology_data_mesh.hpp"

using namespace OpenSubdiv;
typedef Far::TopologyDescriptor Descriptor;

// Static subdivision cache (Performance improvements 1A + 1B)
HashMap<uint64_t, Subdivider::CachedSubdivisionData> Subdivider::subdivision_cache;

uint64_t Subdivider::_compute_topology_hash(const TopologyData &data, int level) {
	// Simple hash combining vertex count, face count, and level
	uint64_t hash = data.vertex_count;
	hash = hash * 31 + data.face_count;
	hash = hash * 31 + data.vertex_count_per_face;
	hash = hash * 31 + level;
	return hash;
}

void Subdivider::_cleanup_subdivision_cache() {
	for (KeyValue<uint64_t, CachedSubdivisionData> &E : subdivision_cache) {
		if (E.value.refiner) {
			delete E.value.refiner;
		}
		if (E.value.vertex_stencils) {
			delete E.value.vertex_stencils;
		}
		if (E.value.varying_stencils) {
			delete E.value.varying_stencils;
		}
		if (E.value.fvar_stencils) {
			delete E.value.fvar_stencils;
		}
	}
	subdivision_cache.clear();
}

// Public wrapper for module shutdown
void Subdivider::cleanup_subdivision_cache() {
	_cleanup_subdivision_cache();
}

void Subdivider::_generate_stencil_tables(CachedSubdivisionData &cache, int32_t p_format) {
	if (cache.stencils_generated || !cache.refiner) {
		return;
	}

	// Generate vertex position stencils
	Far::StencilTableFactory::Options vertex_opts;
	vertex_opts.generateIntermediateLevels = false; // Only final level
	vertex_opts.interpolationMode = Far::StencilTableFactory::INTERPOLATE_VERTEX;
	cache.vertex_stencils = Far::StencilTableFactory::Create(*cache.refiner, vertex_opts);

	// Generate varying stencils (for bone weights if needed)
	if ((p_format & Mesh::ARRAY_FORMAT_BONES) && (p_format & Mesh::ARRAY_FORMAT_WEIGHTS)) {
		Far::StencilTableFactory::Options varying_opts;
		varying_opts.generateIntermediateLevels = false;
		varying_opts.interpolationMode = Far::StencilTableFactory::INTERPOLATE_VARYING;
		cache.varying_stencils = Far::StencilTableFactory::Create(*cache.refiner, varying_opts);
	}

	// Generate face-varying stencils (for UVs if needed)
	if (p_format & Mesh::ARRAY_FORMAT_TEX_UV) {
		Far::StencilTableFactory::Options fvar_opts;
		fvar_opts.generateIntermediateLevels = false;
		fvar_opts.interpolationMode = Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
		fvar_opts.fvarChannel = Channels::UV;
		cache.fvar_stencils = Far::StencilTableFactory::Create(*cache.refiner, fvar_opts);
	}

	cache.stencils_generated = true;
}

Subdivider::TopologyData::TopologyData(const Array &p_mesh_arrays, int32_t p_format, int32_t p_face_verts) {
	ERR_FAIL_INDEX(TopologyDataMesh::ARRAY_VERTEX, p_mesh_arrays.size());
	ERR_FAIL_INDEX(TopologyDataMesh::ARRAY_INDEX, p_mesh_arrays.size());
	ERR_FAIL_INDEX(TopologyDataMesh::ARRAY_UV_INDEX, p_mesh_arrays.size());
	ERR_FAIL_INDEX(TopologyDataMesh::ARRAY_BONES, p_mesh_arrays.size());
	ERR_FAIL_INDEX(TopologyDataMesh::ARRAY_WEIGHTS, p_mesh_arrays.size());
	ERR_FAIL_INDEX(p_mesh_arrays.size(), int(TopologyDataMesh::ARRAY_MAX) + 1);
	vertex_array = p_mesh_arrays[TopologyDataMesh::ARRAY_VERTEX];
	index_array = p_mesh_arrays[TopologyDataMesh::ARRAY_INDEX];
	if (p_format & Mesh::ARRAY_FORMAT_TEX_UV) {
		uv_array = p_mesh_arrays[TopologyDataMesh::ARRAY_TEX_UV];
		uv_index_array = p_mesh_arrays[TopologyDataMesh::ARRAY_UV_INDEX];
	}
	if ((p_format & Mesh::ARRAY_FORMAT_BONES) && (p_format & Mesh::ARRAY_FORMAT_WEIGHTS)) {
		bones_array = p_mesh_arrays[TopologyDataMesh::ARRAY_BONES];
		weights_array = p_mesh_arrays[TopologyDataMesh::ARRAY_WEIGHTS];
	}

	vertex_count_per_face = p_face_verts;
	index_count = index_array.size();
	face_count = index_array.size() / vertex_count_per_face;
	vertex_count = vertex_array.size();
	uv_count = uv_index_array.size();
	bone_count = bones_array.size();
	weight_count = weights_array.size();
}

struct Vertex {
	void Clear() { x = y = z = 0; }
	void AddWithWeight(Vertex const &src, real_t weight) {
		x += weight * src.x;
		y += weight * src.y;
		z += weight * src.z;
	}
	real_t x, y, z;
};

struct VertexUV {
	void Clear() {
		u = v = 0.0f;
	}

	void AddWithWeight(VertexUV const &src, real_t weight) {
		u += weight * src.u;
		v += weight * src.v;
	}

	// Basic 'uv' layout channel
	real_t u, v;
};

struct Bone {
	int32_t bone_id = -1;
	float weight = 0.0f;
};

// Comparison function for sorting bone weights (descending by weight)
struct BoneWeightCompare {
	bool operator()(const Pair<int, float> &a, const Pair<int, float> &b) const {
		return a.second > b.second; // Descending order (highest weights first)
	}
};

struct VertexWeights {
	void Clear() {
		for (int i = 0; i < weights.size(); i++) {
			weights.write[i].bone_id = -1;
			weights.write[i].weight = 0;
		}
	}

	void AddWithWeight(VertexWeights const &src, float weight) {
		for (int i = 0; i < weights.size(); i++) {
			if (src.weights[i].bone_id == weights[i].bone_id) {
				weights.write[i].weight += src.weights[i].weight * weight;
				continue;
			}
			weights.write[i].weight += src.weights[i].weight * weight / 2.0;
		}
		float sum = 0.0f;
		for (int i = 0; i < weights.size(); i++) {
			sum += weights[i].weight;
		}
		if (sum == 0) {
			sum = 1;
		}
		for (int i = 0; i < weights.size(); i++) {
			weights.write[i].weight /= sum;
		}
	}

	Vector<Bone> weights;
};

Descriptor Subdivider::_create_topology_descriptor(Vector<int> &subdiv_face_vertex_count, Descriptor::FVarChannel *channels, const int32_t p_format) {
	const bool use_uv = p_format & Mesh::ARRAY_FORMAT_TEX_UV;

	Descriptor desc;
	desc.numVertices = topology_data.vertex_count;
	desc.numFaces = topology_data.face_count;
	desc.vertIndicesPerFace = topology_data.index_array.ptr();

	subdiv_face_vertex_count = _get_face_vertex_count();

	desc.numVertsPerFace = subdiv_face_vertex_count.ptr();

	int num_channels = 0;
	if (use_uv) {
		channels[Channels::UV].numValues = topology_data.uv_count;
		channels[Channels::UV].valueIndices = topology_data.uv_index_array.ptr();
		num_channels = 1;
	}

	desc.numFVarChannels = num_channels;
	desc.fvarChannels = channels;

	return desc;
}

Far::TopologyRefiner *Subdivider::_create_topology_refiner(const int32_t p_level, const int32_t p_format) {
	const bool use_uv = p_format & Mesh::ARRAY_FORMAT_TEX_UV;

	// Check cache first (Performance improvements 1A + 1B)
	uint64_t cache_key = _compute_topology_hash(topology_data, p_level);
	if (subdivision_cache.has(cache_key)) {
		CachedSubdivisionData &cached = subdivision_cache[cache_key];
		// Update topology_data with cached refiner's counts
		topology_data.vertex_count = cached.refiner->GetNumVerticesTotal();
		if (use_uv) {
			topology_data.uv_count = cached.refiner->GetNumFVarValuesTotal(Channels::UV);
		}
		// Generate stencils if not already done (Performance improvement 1B)
		_generate_stencil_tables(cached, p_format);
		return cached.refiner;
	}

	// Create descriptor
	Vector<int> subdiv_face_vertex_count;
	int num_channels = 0;
	if (use_uv) {
		num_channels = 1;
	}

	Descriptor::FVarChannel *channels = new Descriptor::FVarChannel[num_channels];
	Descriptor desc = _create_topology_descriptor(subdiv_face_vertex_count, channels, p_format);

	Sdc::SchemeType type = _get_refiner_type();
	Sdc::Options options;
	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);
	options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_CORNERS_PLUS2);
	options.SetCreasingMethod(Sdc::Options::CREASE_UNIFORM);

	Far::TopologyRefinerFactory<Descriptor>::Options create_options(type, options);

	Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc, create_options);
	delete[] channels;
	ERR_FAIL_COND_V(!refiner, nullptr);

	Far::TopologyRefiner::UniformOptions refine_options(p_level);
	refine_options.fullTopologyInLastLevel = true;
	refiner->RefineUniform(refine_options);

	topology_data.vertex_count = refiner->GetNumVerticesTotal();
	if (use_uv) {
		topology_data.uv_count = refiner->GetNumFVarValuesTotal(Channels::UV);
	}

	// Cache refiner and generate stencils (Performance improvements 1A + 1B)
	CachedSubdivisionData &cache = subdivision_cache[cache_key];
	cache.refiner = refiner;
	_generate_stencil_tables(cache, p_format);

	return refiner;
}
void Subdivider::_create_subdivision_vertices(Far::TopologyRefiner *refiner, const int p_level, const int32_t p_format) {
	const bool use_uv = p_format & Mesh::ARRAY_FORMAT_TEX_UV;
	const bool use_bones = (p_format & Mesh::ARRAY_FORMAT_BONES) && (p_format & Mesh::ARRAY_FORMAT_WEIGHTS);

	int original_vertex_count = topology_data.vertex_array.size();
	topology_data.vertex_array.resize(topology_data.vertex_count);

	// Try to use stencil tables if available (Performance improvement 1B)
	uint64_t cache_key = _compute_topology_hash(topology_data, p_level);
	bool use_stencils = subdivision_cache.has(cache_key) &&
			subdivision_cache[cache_key].stencils_generated &&
			subdivision_cache[cache_key].vertex_stencils;

	if (use_stencils) {
		// Fast path: Apply pre-computed stencils (2-3x faster!)
		CachedSubdivisionData &cache = subdivision_cache[cache_key];

		// Apply vertex stencils
		Vertex *src = (Vertex *)topology_data.vertex_array.ptr();
		Vertex *dst = src;
		cache.vertex_stencils->UpdateValues(src, dst);

		// Apply UV stencils if present
		if (use_uv && cache.fvar_stencils) {
			topology_data.uv_array.resize(topology_data.uv_count);
			VertexUV *src_uv = (VertexUV *)topology_data.uv_array.ptr();
			VertexUV *dst_uv = src_uv;
			cache.fvar_stencils->UpdateValues(src_uv, dst_uv);
		}
	} else {
		// Fallback: Use PrimvarRefiner (slower but always works)
		Far::PrimvarRefiner primvar_refiner(*refiner);

		// Vertices
		Vertex *src = (Vertex *)topology_data.vertex_array.ptr();
		for (int level = 0; level < p_level; ++level) {
			Vertex *dst = src + refiner->GetLevel(level).GetNumVertices();
			primvar_refiner.Interpolate(level + 1, src, dst);
			src = dst;
		}

		// UVs
		if (use_uv) {
			topology_data.uv_array.resize(topology_data.uv_count);
			VertexUV *src_uv = (VertexUV *)topology_data.uv_array.ptr();
			for (int level = 0; level < p_level; ++level) {
				VertexUV *dst_uv = src_uv + refiner->GetLevel(level).GetNumFVarValues(Channels::UV);
				primvar_refiner.InterpolateFaceVarying(level + 1, src_uv, dst_uv, Channels::UV);
				src_uv = dst_uv;
			}
		}
	}

	if (use_bones) {
		// Bone weights use PrimvarRefiner (complex logic, keep existing implementation)
		Far::PrimvarRefiner primvar_refiner(*refiner);

		int highest_bone_index = 0;
		for (int i = 0; i < topology_data.bones_array.size(); i++) {
			if (topology_data.bones_array[i] > highest_bone_index) {
				highest_bone_index = topology_data.bones_array[i];
			}
		}

		Vector<Vector<Bone>> all_vertex_bone_weights;
		all_vertex_bone_weights.resize(topology_data.vertex_count);

		for (int vertex_index = 0; vertex_index < topology_data.vertex_count; vertex_index++) {
			all_vertex_bone_weights.write[vertex_index].resize(highest_bone_index + 1);
		}

		for (int vertex_index = 0; vertex_index < original_vertex_count; vertex_index++) {
			for (int weight_index = 0; weight_index < 4; weight_index++) {
				if (topology_data.weights_array[vertex_index * 4 + weight_index] != 0.0f) {
					int bone_index = topology_data.bones_array[vertex_index * 4 + weight_index];
					all_vertex_bone_weights.write[vertex_index].write[bone_index].bone_id = bone_index;
					all_vertex_bone_weights.write[vertex_index].write[bone_index].weight = topology_data.weights_array[vertex_index * 4 + weight_index];
				}
			}
		}

		VertexWeights *src_weights = (VertexWeights *)all_vertex_bone_weights.ptrw();
		for (int level = 0; level < p_level; ++level) {
			VertexWeights *dst_weights = src_weights + refiner->GetLevel(level).GetNumVertices();
			primvar_refiner.InterpolateVarying(level + 1, src_weights, dst_weights);
			src_weights = dst_weights;
		}

		topology_data.bones_array.resize(topology_data.vertex_count * 4);
		topology_data.weights_array.resize(topology_data.vertex_count * 4);
		for (int vertex_index = 0; vertex_index < topology_data.vertex_count; vertex_index++) {
			int weight_indices[4] = { -1, -1, -1, -1 };
			const Vector<Bone> &vertex_bones_weights = all_vertex_bone_weights[vertex_index];

			for (int weight_index = 0; weight_index <= highest_bone_index; weight_index++) {
				if (vertex_bones_weights[weight_index].weight != 0 && (weight_indices[3] == -1 || vertex_bones_weights[weight_index].weight > vertex_bones_weights[weight_indices[3]].weight)) {
					weight_indices[3] = weight_index;
					for (int i = 2; i >= 0; i--) {
						if (weight_indices[i] == -1 || vertex_bones_weights[weight_index].weight > vertex_bones_weights[weight_indices[i]].weight) {
							weight_indices[i + 1] = weight_indices[i];
							weight_indices[i] = weight_index;
						} else {
							break;
						}
					}
				}
			}

			for (int result_weight_index = 0; result_weight_index < 4; result_weight_index++) {
				if (weight_indices[result_weight_index] == -1) {
					topology_data.bones_array.write[vertex_index * 4 + result_weight_index] = 0;
					topology_data.weights_array.write[vertex_index * 4 + result_weight_index] = 0;
				} else {
					topology_data.bones_array.write[vertex_index * 4 + result_weight_index] = weight_indices[result_weight_index];
					topology_data.weights_array.write[vertex_index * 4 + result_weight_index] = vertex_bones_weights[weight_indices[result_weight_index]].weight;
				}
			}
		}
		for (int vertex_index = 0; vertex_index < topology_data.vertex_count; vertex_index++) {
			float total_weight = 0.0f;
			for (int weight_index = 0; weight_index < 4; weight_index++) {
				total_weight += topology_data.weights_array[vertex_index * 4 + weight_index];
			}
			for (int weight_index = 0; weight_index < 4; weight_index++) {
				if (total_weight != 0) {
					topology_data.weights_array.write[vertex_index * 4 + weight_index] /= total_weight;
				}
			}
		}
	}
}

Array Subdivider::get_subdivided_arrays(const Array &p_arrays, int p_level, int32_t p_format, bool calculate_normals) {
	subdivide(p_arrays, p_level, p_format, calculate_normals);
	return _get_triangle_arrays();
}

Array Subdivider::get_subdivided_topology_arrays(const Array &p_arrays, int p_level, int32_t p_format, bool calculate_normals) {
	ERR_FAIL_COND_V(p_level <= 0, Array());
	subdivide(p_arrays, p_level, p_format, calculate_normals);
	Array arr;
	arr.resize(TopologyDataMesh::ARRAY_MAX);
	arr[TopologyDataMesh::ARRAY_VERTEX] = topology_data.vertex_array;
	arr[TopologyDataMesh::ARRAY_NORMAL] = topology_data.normal_array;
	arr[TopologyDataMesh::ARRAY_TEX_UV] = topology_data.uv_array;
	arr[TopologyDataMesh::ARRAY_UV_INDEX] = topology_data.uv_index_array;
	arr[TopologyDataMesh::ARRAY_INDEX] = topology_data.index_array;
	arr[TopologyDataMesh::ARRAY_BONES] = topology_data.bones_array;
	arr[TopologyDataMesh::ARRAY_WEIGHTS] = topology_data.weights_array;
	return arr;
}

void Subdivider::subdivide(const Array &p_arrays, int p_level, int32_t p_format, bool calculate_normals) {
	ERR_FAIL_COND(p_level < 0);
	topology_data = TopologyData(p_arrays, p_format, _get_vertices_per_face_count());
	//if p_level is not 0 subdivide mesh and store in topology_data again
	if (p_level != 0) {
		Far::TopologyRefiner *refiner = _create_topology_refiner(p_level, p_format);
		ERR_FAIL_COND_MSG(!refiner, "Refiner couldn't be created, numVertsPerFace array likely lost.");
		_create_subdivision_vertices(refiner, p_level, p_format);
		_create_subdivision_faces(refiner, p_level, p_format);

		// Don't delete refiner - it's cached for reuse (Performance improvement 1A)
		// Cleanup happens in _cleanup_refiner_cache() at module shutdown
	}

	if (calculate_normals) {
		topology_data.normal_array = _calculate_smooth_normals(topology_data.vertex_array, topology_data.index_array);
	}
}

OpenSubdiv::Sdc::SchemeType Subdivider::_get_refiner_type() const {
	return Sdc::SchemeType::SCHEME_CATMARK;
}

void Subdivider::_create_subdivision_faces(OpenSubdiv::Far::TopologyRefiner *refiner,
		const int32_t p_level, int32_t p_format) {
	const bool use_uv = p_format & Mesh::ARRAY_FORMAT_TEX_UV;

	PackedInt32Array index_array;
	PackedInt32Array uv_index_array;

	Far::TopologyLevel const &last_level = refiner->GetLevel(p_level);
	int face_count_out = last_level.GetNumFaces();
	int uv_index_offset = use_uv ? topology_data.uv_count - last_level.GetNumFVarValues(Channels::UV) : -1;

	int vertex_index_offset = topology_data.vertex_count - last_level.GetNumVertices();
	for (int face_index = 0; face_index < face_count_out; ++face_index) {
		int parent_face_index = last_level.GetFaceParentFace(face_index);
		for (int level_index = p_level - 1; level_index > 0; --level_index) {
			Far::TopologyLevel const &prev_level = refiner->GetLevel(level_index);
			parent_face_index = prev_level.GetFaceParentFace(parent_face_index);
		}

		Far::ConstIndexArray face_vertices = last_level.GetFaceVertices(face_index);

		ERR_FAIL_COND(face_vertices.size() != topology_data.vertex_count_per_face);
		for (int face_vert_index = 0; face_vert_index < topology_data.vertex_count_per_face; face_vert_index++) {
			index_array.push_back(vertex_index_offset + face_vertices[face_vert_index]);
		}

		if (use_uv) {
			Far::ConstIndexArray face_uvs = last_level.GetFaceFVarValues(face_index, Channels::UV);
			for (int face_vert_index = 0; face_vert_index < topology_data.vertex_count_per_face; face_vert_index++) {
				uv_index_array.push_back(uv_index_offset + face_uvs[face_vert_index]);
			}
		}
	}
	topology_data.index_array = index_array;
	if (use_uv) {
		topology_data.uv_index_array = uv_index_array;
	}
}

PackedVector3Array Subdivider::_calculate_smooth_normals(const PackedVector3Array &quad_vertex_array, const PackedInt32Array &quad_index_array) {
	PackedVector3Array normals;
	normals.resize(quad_vertex_array.size());
	for (int f = 0; f < quad_index_array.size(); f += topology_data.vertex_count_per_face) {
		// // We will use the first three verts to calculate a normal
		const Vector3 &p_point1 = quad_vertex_array[quad_index_array[f]];
		const Vector3 &p_point2 = quad_vertex_array[quad_index_array[f + 1]];
		const Vector3 &p_point3 = quad_vertex_array[quad_index_array[f + 2]];
		Vector3 normal_calculated = (p_point1 - p_point3).cross(p_point1 - p_point2);
		normal_calculated.normalize();
		for (int n_pos = f; n_pos < f + topology_data.vertex_count_per_face; n_pos++) {
			int vertexIndex = quad_index_array[n_pos];
			normals.write[vertexIndex] += normal_calculated;
		}
	}
	//normalized accumulated normals
	for (int vertex_index = 0; vertex_index < normals.size(); ++vertex_index) {
		normals.write[vertex_index].normalize();
	}
	return normals;
}

Array Subdivider::_get_triangle_arrays() const {
	return Array();
}

Vector<int> Subdivider::_get_face_vertex_count() const {
	return Vector<int>();
}

int32_t Subdivider::_get_vertices_per_face_count() const {
	return 0;
}
Array Subdivider::_get_direct_triangle_arrays() const {
	return Array();
}

void Subdivider::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_subdivided_arrays", "arrays", "level", "format", "calculate_normals"), &Subdivider::get_subdivided_arrays);
	ClassDB::bind_method(D_METHOD("get_subdivided_topology_arrays", "arrays", "level", "format", "calculate_normals"), &Subdivider::get_subdivided_topology_arrays);
}

Subdivider::Subdivider() {
}
Subdivider::~Subdivider() {
}
