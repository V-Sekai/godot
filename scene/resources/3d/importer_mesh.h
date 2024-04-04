/**************************************************************************/
/*  importer_mesh.h                                                       */
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

#ifndef IMPORTER_MESH_H
#define IMPORTER_MESH_H

#include "core/io/resource.h"
#include "core/templates/local_vector.h"
#include "scene/resources/3d/concave_polygon_shape_3d.h"
#include "scene/resources/3d/convex_polygon_shape_3d.h"
#include "scene/resources/mesh.h"
#include "scene/resources/navigation_mesh.h"

#include <cstdint>

// The following classes are used by importers instead of ArrayMesh and MeshInstance3D
// so the data is not registered (hence, quality loss), importing happens faster and
// its easier to modify before saving

class ImporterMesh : public Resource {
	GDCLASS(ImporterMesh, Resource)

	struct Surface {
		Mesh::PrimitiveType primitive;
		Array arrays;
		struct BlendShape {
			Array arrays;
		};
		Vector<BlendShape> blend_shape_data;
		struct LOD {
			Vector<int> indices;
			float distance = 0.0f;
		};
		Vector<LOD> lods;
		Ref<Material> material;
		String name;
		uint64_t flags = 0;

		struct LODComparator {
			_FORCE_INLINE_ bool operator()(const LOD &l, const LOD &r) const {
				return l.distance < r.distance;
			}
		};

		void split_normals(const LocalVector<int> &p_indices, const LocalVector<Vector3> &p_normals);
		static void _split_normals(Array &r_arrays, const LocalVector<int> &p_indices, const LocalVector<Vector3> &p_normals);
	};
	Vector<Surface> surfaces;
	Vector<String> blend_shapes;
	Mesh::BlendShapeMode blend_shape_mode = Mesh::BLEND_SHAPE_MODE_NORMALIZED;

	Ref<ArrayMesh> mesh;

	Ref<ImporterMesh> shadow_mesh;

	Size2i lightmap_size_hint;

	struct MergedAttribute {
		Vector3 normal;
		Vector<int> primary_bone_influence;
	};

	Vector<float> merged_attribute_to_float32_array(const MergedAttribute *p_attributes, size_t p_count) {
		Vector<float> floats;
		if (p_count == 0) {
			return floats;
		}

		size_t total_size = 0;
		for (size_t i = 0; i < p_count; ++i) {
			total_size += 3 + p_attributes[i].primary_bone_influence.size();
		}

		floats.resize(total_size);
		float *floats_w = floats.ptrw();

		for (size_t i = 0; i < p_count; ++i) {
			const MergedAttribute &attr = p_attributes[i];
			floats_w[0] = attr.normal.x;
			floats_w[1] = attr.normal.y;
			floats_w[2] = attr.normal.z;

			for (int j = 0; j < attr.primary_bone_influence.size(); ++j) {
				floats_w[j + 3] = static_cast<float>(attr.primary_bone_influence[j]);
			}

			floats_w += 3 + attr.primary_bone_influence.size();
		}
		return floats;
	}

	struct PairHasher {
		static _FORCE_INLINE_ uint32_t hash(const Pair<Vector<int>, Vector<int>> &p_pair) {
			uint32_t hash = 0;
			for (int i : p_pair.first) {
				hash = hash * 31 + HashMapHasherDefault::hash(i);
			}
			for (int i : p_pair.second) {
				hash = hash * 31 + HashMapHasherDefault::hash(i);
			}
			return hash;
		}
	};

	struct VertexSimilarityComparator {
		Vector<int> target_bones;

		VertexSimilarityComparator(const Vector<int> &p_target) :
				target_bones(p_target) {}

		_FORCE_INLINE_ bool operator()(const Vector<int> &p_p, const Vector<int> &p_q) const {
			return get_similarity(p_p, target_bones) > get_similarity(p_q, target_bones);
		}

		int get_similarity(const Vector<int> &p_a, const Vector<int> &p_b) const {
			int similarity = 0;
			for (int i : p_a) {
				if (p_b.find(i) != -1) {
					similarity++;
				}
			}
			return similarity;
		}
	};

	struct BoneWeightComparator {
		_FORCE_INLINE_ bool operator()(const Pair<int, float> &p, const Pair<int, float> &q) const {
			if (p.second == q.second) {
				return p.first < q.first; // Use bone index as the tie-breaker.
			}
			return p.second < q.second;
		}
	};

	class TopElements {
	private:
		Vector<Pair<int, float>> elements;
		int limit;

	public:
		TopElements(int p_limit) :
				limit(p_limit) {}

		void insert(Pair<int, float> p_pair) {
			if (elements.size() < limit || BoneWeightComparator()(p_pair, elements[0])) {
				if (elements.size() == limit) {
					elements.remove_at(0);
					for (int i = elements.size() / 2 - 1; i >= 0; i--) {
						heapify(i);
					}
				}

				elements.push_back(p_pair);
				int i = elements.size() - 1;

				while (i > 0 && BoneWeightComparator()(elements[i], elements[(i - 1) / 2])) {
					SWAP(elements.write[i], elements.write[(i - 1) / 2]);
					i = (i - 1) / 2;
				}
			}
		}

		void heapify(int p_index) {
			int smallest = p_index, left = 2 * p_index + 1, right = 2 * p_index + 2;

			if (left < elements.size() && !BoneWeightComparator()(elements[smallest], elements[left])) {
				smallest = left;
			}
			if (right < elements.size() && !BoneWeightComparator()(elements[smallest], elements[right])) {
				smallest = right;
			}

			if (smallest != p_index) {
				SWAP(elements.write[p_index], elements.write[smallest]);
				heapify(smallest);
			}
		}

		Vector<Pair<int, float>> get_elements() {
			Vector<Pair<int, float>> sorted_elements = elements;
			sorted_elements.sort_custom<BoneWeightComparator>(BoneWeightComparator());
			return sorted_elements;
		}
	};

	Vector<int> _get_primary_bone_influence(Vector<int> &r_bones, Vector<float> &r_weights, int p_index) {
		TopElements bone_weight_pairs(8);
		for (int i = 0; i < r_bones.size(); i++) {
			bone_weight_pairs.insert(Pair<int, float>(r_bones[i], r_weights[i]));
		}

		Vector<int> primary_bones;
		for (Pair<int, float> pair : bone_weight_pairs.get_elements()) {
			primary_bones.push_back(pair.first);
		}

		return primary_bones;
	}

	float get_bone_influence_similarity(const Vector<int> &p_influence_1, const Vector<int> &p_influence_2) {
		ERR_FAIL_COND_V(p_influence_1.size() != p_influence_2.size(), 0.0f);

		float dot = 0.0f, denom_a = 0.0f, denom_b = 0.0f;
		for (int i = 0; i < p_influence_1.size(); ++i) {
			dot += p_influence_1[i] * p_influence_2[i];
			denom_a += p_influence_1[i] * p_influence_1[i];
			denom_b += p_influence_2[i] * p_influence_2[i];
		}
		return dot / (sqrt(denom_a) * sqrt(denom_b));
	}

protected:
	void _set_data(const Dictionary &p_data);
	Dictionary _get_data() const;

	static void _bind_methods();

public:
	void add_blend_shape(const String &p_name);
	int get_blend_shape_count() const;
	String get_blend_shape_name(int p_blend_shape) const;

	void add_surface(Mesh::PrimitiveType p_primitive, const Array &p_arrays, const TypedArray<Array> &p_blend_shapes = Array(), const Dictionary &p_lods = Dictionary(), const Ref<Material> &p_material = Ref<Material>(), const String &p_name = String(), const uint64_t p_flags = 0);
	int get_surface_count() const;

	void set_blend_shape_mode(Mesh::BlendShapeMode p_blend_shape_mode);
	Mesh::BlendShapeMode get_blend_shape_mode() const;

	Mesh::PrimitiveType get_surface_primitive_type(int p_surface);
	String get_surface_name(int p_surface) const;
	void set_surface_name(int p_surface, const String &p_name);
	Array get_surface_arrays(int p_surface) const;
	Array get_surface_blend_shape_arrays(int p_surface, int p_blend_shape) const;
	int get_surface_lod_count(int p_surface) const;
	Vector<int> get_surface_lod_indices(int p_surface, int p_lod) const;
	float get_surface_lod_size(int p_surface, int p_lod) const;
	Ref<Material> get_surface_material(int p_surface) const;
	uint64_t get_surface_format(int p_surface) const;

	void set_surface_material(int p_surface, const Ref<Material> &p_material);

	void generate_lods(float p_normal_merge_angle, float p_normal_split_angle, Array p_skin_pose_transform_array);

	void create_shadow_mesh();
	Ref<ImporterMesh> get_shadow_mesh() const;

	Vector<Face3> get_faces() const;
	Vector<Ref<Shape3D>> convex_decompose(const Ref<MeshConvexDecompositionSettings> &p_settings) const;
	Ref<ConvexPolygonShape3D> create_convex_shape(bool p_clean = true, bool p_simplify = false) const;
	Ref<ConcavePolygonShape3D> create_trimesh_shape() const;
	Ref<NavigationMesh> create_navigation_mesh();
	Error lightmap_unwrap_cached(const Transform3D &p_base_transform, float p_texel_size, const Vector<uint8_t> &p_src_cache, Vector<uint8_t> &r_dst_cache);

	void set_lightmap_size_hint(const Size2i &p_size);
	Size2i get_lightmap_size_hint() const;

	bool has_mesh() const;
	Ref<ArrayMesh> get_mesh(const Ref<ArrayMesh> &p_base = Ref<ArrayMesh>());
	void clear();
};

#endif // IMPORTER_MESH_H
