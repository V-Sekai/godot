// Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.
// Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// -- Godot Engine <https://godotengine.org>

#include "editor_scene_importer_alembic.h"
#include "core/bind/core_bind.h"
#include "core/io/image_loader.h"
#include "editor/editor_file_system.h"
#include "editor/import/resource_importer_scene.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/animation/animation_player.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"
#include "zutil.h"
#include <string>

uint32_t EditorSceneImporterAlembic::get_import_flags() const {
	return IMPORT_SCENE;
}

void EditorSceneImporterAlembic::_bind_methods() {
}

Node *EditorSceneImporterAlembic::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, uint32_t p_compress_flags, List<String> *r_missing_deps, Error *r_err) {
	std::wstring w_path = ProjectSettings::get_singleton()->globalize_path(p_path).c_str();
	std::string s_path(w_path.begin(), w_path.end());

	// Referenced http://jonmacey.blogspot.com/2011/12/getting-started-with-alembic.html
	IArchive archive(Alembic::AbcCoreOgawa::ReadArchive(), s_path);

	// Based on AbcTree/AbcTree.cpp

	std::vector<std::string> seglist;
	bool opt_all = false;

	// walk object hierarchy and find valid objects
	AbcG::IObject test = archive.getTop();
	AbcG::IObject iObj = test;
	while (test.valid() && seglist.size() > 0) {
		test = test.getChild(seglist.front());
		if (test.valid()) {
			iObj = test;
			seglist.erase(seglist.begin());
		}
	}

	// walk property hierarchy for most recent valid object
	Abc::ICompoundProperty props = iObj.getProperties();
	const Abc::PropertyHeader *header;
	bool found = false;
	for (std::size_t i = 0; i < seglist.size(); ++i) {
		header = props.getPropertyHeader(seglist[i]);
		if (header && header->isCompound()) {
			Abc::ICompoundProperty ptest(props, header->getName());
			if (ptest.valid()) {
				props = ptest;
				found = true;
			}
		} else if (header && header->isSimple()) {
			found = true;
		} else {
			ERR_FAIL_V_MSG(vformat("Invalid object or property %s", seglist[i]));
		}
	}
	Node3D *root = memnew(Node3D);
	// walk the archive tree
	if (found) {
		if (header->isCompound()) {
			tree(props, root);
		} else {
			tree(Abc::IScalarProperty(props, header->getName()), root);
		}
	} else {
		tree(iObj, root, root, opt_all);
	}
	if (anim.is_valid()) {
		AnimationPlayer *ap = memnew(AnimationPlayer);
		ap->add_animation(test.getName().data(), anim);
		root->add_child(ap);
		ap->set_owner(root);
		anim->set_length(archive.getMaxNumSamplesForTimeSamplingIndex(1));
	}
}

void EditorSceneImporterAlembic::_insert_pivot_anim_track(const Vector<MeshInstance3D *> p_meshes, const String p_node_name, Vector<const aiNodeAnim *> F, AnimationPlayer *ap, Skeleton3D *sk, float &length, float ticks_per_second, Ref<Animation> animation, int p_bake_fps, const String &p_path, const Node *p_scene) {
	NodePath node_path;
	if (sk != NULL) {
		const String path = ap->get_owner()->get_path_to(sk);
		if (path.empty()) {
			return;
		}
		if (sk->find_bone(p_node_name) == -1) {
			return;
		}
		node_path = path + ":" + p_node_name;
		Node *node = sk->get_parent();
		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(node);
	} else {
		Node *node = ap->get_owner()->find_node(p_node_name);
		if (node == NULL) {
			return;
		}
		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(node);
		const String path = ap->get_owner()->get_path_to(node);
		node_path = path;
	}
	if (node_path.is_empty()) {
		return;
	}

	Vector<Vector3> pos_values;
	Vector<float> pos_times;
	Vector<Vector3> scale_values;
	Vector<float> scale_times;
	Vector<Quat> rot_values;
	Vector<float> rot_times;
	Vector3 base_pos;
	Quat base_rot;
	Vector3 base_scale = Vector3(1, 1, 1);
	bool is_translation = false;
	bool is_rotation = false;
	bool is_scaling = false;
	for (size_t k = 0; k < F.size(); k++) {
		String p_track_type = F[k]->mNodeName.data();
		if (p_track_type == "_Translation") {
			is_translation = is_translation || true;
		} else if (p_track_type == "_Rotation") {
			is_rotation = is_rotation || true;
		} else if (p_track_type == "_Scaling") {
			is_scaling = is_scaling || true;
		} else {
			continue;
		}
		ERR_CONTINUE(ap->get_owner()->has_node(node_path) == false);

		if (F[k]->mNumRotationKeys || F[k]->mNumPositionKeys || F[k]->mNumScalingKeys) {
			if (is_rotation) {
				for (int i = 0; i < F[k]->mNumRotationKeys; i++) {
					length = MAX(length, F[k]->mRotationKeys[i].mTime / ticks_per_second);
				}
			}
			if (is_translation) {
				for (int i = 0; i < F[k]->mNumPositionKeys; i++) {
					length = MAX(length, F[k]->mPositionKeys[i].mTime / ticks_per_second);
				}
			}
			if (is_scaling) {
				for (int i = 0; i < F[k]->mNumScalingKeys; i++) {
					length = MAX(length, F[k]->mScalingKeys[i].mTime / ticks_per_second);
				}
			}

			if (is_rotation == false && is_translation == false && is_scaling == false) {
				return;
			}

			if (is_rotation) {
				if (F[k]->mNumRotationKeys != 0) {
					aiQuatKey key = F[k]->mRotationKeys[0];
					real_t x = key.mValue.x;
					real_t y = key.mValue.y;
					real_t z = key.mValue.z;
					real_t w = key.mValue.w;
					Quat q(x, y, z, w);
					q = q.normalized();
					base_rot = q;
				}
			}

			if (is_translation) {
				if (F[k]->mNumPositionKeys != 0) {
					aiVectorKey key = F[k]->mPositionKeys[0];
					real_t x = key.mValue.x;
					real_t y = key.mValue.y;
					real_t z = key.mValue.z;
					base_pos = Vector3(x, y, z);
				}
			}

			if (is_scaling) {
				if (F[k]->mNumScalingKeys != 0) {
					aiVectorKey key = F[k]->mScalingKeys[0];
					real_t x = key.mValue.x;
					real_t y = key.mValue.y;
					real_t z = key.mValue.z;
					base_scale = Vector3(x, y, z);
				}
			}
			if (is_translation) {
				for (size_t p = 0; p < F[k]->mNumPositionKeys; p++) {
					aiVector3D pos = F[k]->mPositionKeys[p].mValue;
					pos_values.push_back(Vector3(pos.x, pos.y, pos.z));
					pos_times.push_back(F[k]->mPositionKeys[p].mTime / ticks_per_second);
				}
			}

			if (is_rotation) {
				for (size_t r = 0; r < F[k]->mNumRotationKeys; r++) {
					aiQuaternion quat = F[k]->mRotationKeys[r].mValue;
					rot_values.push_back(Quat(quat.x, quat.y, quat.z, quat.w).normalized());
					rot_times.push_back(F[k]->mRotationKeys[r].mTime / ticks_per_second);
				}
			}

			if (is_scaling) {
				for (size_t sc = 0; sc < F[k]->mNumScalingKeys; sc++) {
					aiVector3D scale = F[k]->mScalingKeys[sc].mValue;
					scale_values.push_back(Vector3(scale.x, scale.y, scale.z));
					scale_times.push_back(F[k]->mScalingKeys[sc].mTime / ticks_per_second);
				}
			}
		}
	}
	int32_t track_idx = animation->get_track_count();
	animation->add_track(Animation::TYPE_TRANSFORM);
	animation->track_set_path(track_idx, node_path);
	float increment = 1.0 / float(p_bake_fps);
	float time = 0.0;
	bool last = false;
	while (true) {
		Vector3 pos = Vector3();
		Quat rot = Quat();
		Vector3 scale = Vector3(1.0f, 1.0f, 1.0f);
		if (is_translation && pos_values.size()) {
			pos = _interpolate_track<Vector3>(pos_times, pos_values, time, AssetImportAnimation::INTERP_LINEAR);
			Transform3D anim_xform;
			String ext = p_path.get_file().get_extension().to_lower();
		}
		if (is_rotation && rot_values.size()) {
			rot = _interpolate_track<Quat>(rot_times, rot_values, time, AssetImportAnimation::INTERP_LINEAR).normalized();
		}
		if (is_scaling && scale_values.size()) {
			scale = _interpolate_track<Vector3>(scale_times, scale_values, time, AssetImportAnimation::INTERP_LINEAR);
		}
		animation->track_set_interpolation_type(track_idx, Animation::INTERPOLATION_LINEAR);
		animation->transform_track_insert_key(track_idx, time, pos, rot, scale);

		if (last) {
			break;
		}
		time += increment;
		if (time >= length) {
			last = true;
			time = length;
		}
	}
}

void EditorSceneImporterAlembic::_add_mesh_to_mesh_instance(const aiNode *p_node, const Node *p_scene, Skeleton3D *s, const String &p_path, MeshInstance3D *p_mesh_instance, Node *p_owner, HashSet<String> &r_bone_name, int32_t &r_mesh_count, int32_t p_max_bone_weights) {
	Ref<ArrayMesh> mesh;
	mesh.instance();
	bool has_uvs = false;
	for (size_t i = 0; i < p_node->mNumMeshes; i++) {
		const unsigned int mesh_idx = p_node->mMeshes[i];
		const aiMesh *ai_mesh = p_scene->mMeshes[mesh_idx];

		HashMap<uint32_t, Vector<float>> vertex_weight;
		HashMap<uint32_t, Vector<String>> vertex_bone_name;

		Ref<SurfaceTool> st;
		st.instance();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);

		for (size_t j = 0; j < ai_mesh->mNumVertices; j++) {
			if (ai_mesh->HasTextureCoords(0)) {
				has_uvs = true;
				st->add_uv(Vector2(ai_mesh->mTextureCoords[0][j].x, 1.0f - ai_mesh->mTextureCoords[0][j].y));
			}
			if (ai_mesh->HasTextureCoords(1)) {
				has_uvs = true;
				st->add_uv2(Vector2(ai_mesh->mTextureCoords[1][j].x, 1.0f - ai_mesh->mTextureCoords[1][j].y));
			}
			if (ai_mesh->HasVertexColors(0)) {
				Color color = Color(ai_mesh->mColors[0]->r, ai_mesh->mColors[0]->g, ai_mesh->mColors[0]->b, ai_mesh->mColors[0]->a);
				st->add_color(color);
			}
			if (ai_mesh->mNormals != NULL) {
				const aiVector3D normals = ai_mesh->mNormals[j];
				const Vector3 godot_normal = Vector3(normals.x, normals.y, normals.z);
				st->add_normal(godot_normal);
				if (ai_mesh->HasTangentsAndBitangents()) {
					const aiVector3D tangents = ai_mesh->mTangents[j];
					const Vector3 godot_tangent = Vector3(tangents.x, tangents.y, tangents.z);
					const aiVector3D bitangent = ai_mesh->mBitangents[j];
					const Vector3 godot_bitangent = Vector3(bitangent.x, bitangent.y, bitangent.z);
					float d = godot_normal.cross(godot_tangent).dot(godot_bitangent) > 0.0f ? 1.0f : -1.0f;
					st->add_tangent(Plane(tangents.x, tangents.y, tangents.z, d));
				}
			}

			if (s != NULL && s->get_bone_count() > 0) {
				HashMap<uint32_t, Vector<String>>::Element *I = vertex_bone_name.find(j);
				Vector<int32_t> bones;
				if (I != NULL) {
					Vector<String> bone_names;
					bone_names.append_array(I->value());
					for (size_t f = 0; f < bone_names.size(); f++) {
						int32_t bone = s->find_bone(bone_names[f]);
						ERR_PRINT("Alembic Importer: Mesh can't find bone " + bone_names[f]);
						ERR_FAIL_COND(bone == -1);
						bones.push_back(bone);
					}
					if (s->get_bone_count()) {
						int32_t add = CLAMP(p_max_bone_weights - bones.size(), 0, p_max_bone_weights);
						for (size_t f = 0; f < add; f++) {
							bones.push_back(0);
						}
					}
					st->add_bones(bones);
					HashMap<uint32_t, Vector<float>>::Element *E = vertex_weight.find(j);
					Vector<float> weights;
					if (E != NULL) {
						weights = E->value();
						if (weights.size() != p_max_bone_weights) {
							int32_t add = CLAMP(p_max_bone_weights - weights.size(), 0, p_max_bone_weights);
							for (size_t f = 0; f < add; f++) {
								weights.push_back(0.0f);
							}
						}
					}
					ERR_CONTINUE(weights.size() == 0);
					st->add_weights(weights);
				}
			}
			const aiVector3D pos = ai_mesh->mVertices[j];
			Vector3 godot_pos = Vector3(pos.x, pos.y, pos.z);
			st->add_vertex(godot_pos);
		}
		for (size_t j = 0; j < ai_mesh->mNumFaces; j++) {
			const aiFace face = ai_mesh->mFaces[j];
			ERR_FAIL_COND(face.mNumIndices != 3);
			Vector<size_t> order;
			order.push_back(2);
			order.push_back(1);
			order.push_back(0);
			for (size_t k = 0; k < order.size(); k++) {
				ERR_FAIL_COND(face.mIndices[order[k]] >= st->get_vertex_array().size());
				st->add_index(face.mIndices[order[k]]);
			}
		}
		if (ai_mesh->HasTangentsAndBitangents() == false && has_uvs) {
			st->generate_tangents();
		}
		aiMaterial *ai_material = p_scene->mMaterials[ai_mesh->mMaterialIndex];
		Ref<SpatialMaterial> mat;
		mat.instance();

		int32_t mat_two_sided = 0;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_TWOSIDED, mat_two_sided)) {
			if (mat_two_sided > 0) {
				mat->set_cull_mode(SpatialMaterial::CULL_DISABLED);
			}
		}

		const String mesh_name = _ai_string_to_string(ai_mesh->mName);
		aiString mat_name;
		if (AI_SUCCESS == ai_material->Get(AI_MATKEY_NAME, mat_name)) {
			mat->set_name(_ai_string_to_string(mat_name));
		}

		aiTextureType tex_emissive = aiTextureType_EMISSIVE;

		aiTextureType tex_albedo = aiTextureType_DIFFUSE;
		if (ai_material->GetTextureCount(tex_albedo) > 0) {
			aiString ai_filename = aiString();
			String filename = "";
			aiTextureMapMode map_mode[2];
			if (AI_SUCCESS == ai_material->GetTexture(tex_albedo, 0, &ai_filename, NULL, NULL, NULL, NULL, map_mode)) {
				filename = _ai_raw_string_to_string(ai_filename);
				String path = p_path.get_base_dir() + "/" + filename.replace("\\", "/");
				bool found = false;
				_find_texture_path(p_path, path, found);
				if (found) {
					Ref<Texture> texture = _load_texture(p_scene, path);
					if (texture != NULL) {
						if (texture->get_data()->detect_alpha() != Image::ALPHA_NONE) {
							_set_texture_mapping_mode(map_mode, texture);
							mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
							mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
						}
						mat->set_texture(SpatialMaterial::TEXTURE_ALBEDO, texture);
					}
				}
			}
		} else {
			aiColor4D clr_diffuse;
			if (AI_SUCCESS == ai_material->Get(AI_MATKEY_COLOR_DIFFUSE, clr_diffuse)) {
				if (Math::is_equal_approx(real_t(clr_diffuse.a), real_t(1.0)) == false) {
					mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
					mat->set_depth_draw_mode(SpatialMaterial::DepthDrawMode::DEPTH_DRAW_ALPHA_OPAQUE_PREPASS);
				}
				mat->set_albedo(Color(clr_diffuse.r, clr_diffuse.g, clr_diffuse.b, clr_diffuse.a));
			}
		}

		aiString cull_mode;
		if (p_node->mMetaData) {
			p_node->mMetaData->Get("Culling", cull_mode);
		}
		if (cull_mode.length != 0 && cull_mode == aiString("CullingOff")) {
			mat->set_cull_mode(SpatialMaterial::CULL_DISABLED);
		}

		Array array_mesh = st->commit_to_arrays();
		Array morphs;
		morphs.resize(ai_mesh->mNumAnimMeshes);
		Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
		HashMap<uint32_t, String> morph_mesh_idx_names;
		for (int i = 0; i < ai_mesh->mNumAnimMeshes; i++) {
			String ai_anim_mesh_name = _ai_string_to_string(ai_mesh->mAnimMeshes[i]->mName);
			mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);
			if (ai_anim_mesh_name.empty()) {
				ai_anim_mesh_name = String("morph_") + itos(i);
			}
			mesh->add_blend_shape(ai_anim_mesh_name);
			morph_mesh_idx_names.insert(i, ai_anim_mesh_name);
			Array array_copy;
			array_copy.resize(VisualServer::ARRAY_MAX);

			for (int l = 0; l < VisualServer::ARRAY_MAX; l++) {
				array_copy[l] = array_mesh[l].duplicate(true);
			}

			const uint32_t num_vertices = ai_mesh->mAnimMeshes[i]->mNumVertices;
			array_copy[Mesh::ARRAY_INDEX] = Variant();
			if (ai_mesh->mAnimMeshes[i]->HasPositions()) {
				PoolVector3Array vertices;
				vertices.resize(num_vertices);
				for (int l = 0; l < num_vertices; l++) {
					const aiVector3D ai_pos = ai_mesh->mAnimMeshes[i]->mVertices[l];
					Vector3 position = Vector3(ai_pos.x, ai_pos.y, ai_pos.z);
					vertices.write()[l] = position;
				}
				PoolVector3Array new_vertices = array_copy[VisualServer::ARRAY_VERTEX].duplicate(true);

				for (int l = 0; l < vertices.size(); l++) {
					PoolVector3Array::Write w = new_vertices.write();
					w[l] = vertices[l];
				}
				ERR_CONTINUE(vertices.size() != new_vertices.size());
				array_copy[VisualServer::ARRAY_VERTEX] = new_vertices;
			}

			int32_t color_set = 0;
			if (ai_mesh->mAnimMeshes[i]->HasVertexColors(color_set)) {
				PoolColorArray colors;
				colors.resize(num_vertices);
				for (int l = 0; l < num_vertices; l++) {
					const aiColor4D ai_color = ai_mesh->mAnimMeshes[i]->mColors[color_set][l];
					Color color = Color(ai_color.r, ai_color.g, ai_color.b, ai_color.a);
					colors.write()[l] = color;
				}
				PoolColorArray new_colors = array_copy[VisualServer::ARRAY_COLOR].duplicate(true);

				for (int l = 0; l < colors.size(); l++) {
					PoolColorArray::Write w = new_colors.write();
					w[l] = colors[l];
				}
				array_copy[VisualServer::ARRAY_COLOR] = new_colors;
			}

			if (ai_mesh->mAnimMeshes[i]->HasNormals()) {
				PoolVector3Array normals;
				normals.resize(num_vertices);
				for (int l = 0; l < num_vertices; l++) {
					const aiVector3D ai_normal = ai_mesh->mAnimMeshes[i]->mNormals[l];
					Vector3 normal = Vector3(ai_normal.x, ai_normal.y, ai_normal.z);
					normals.write()[l] = normal;
				}
				PoolVector3Array new_normals = array_copy[VisualServer::ARRAY_NORMAL].duplicate(true);

				for (int l = 0; l < normals.size(); l++) {
					PoolVector3Array::Write w = new_normals.write();
					w[l] = normals[l];
				}
				array_copy[VisualServer::ARRAY_NORMAL] = new_normals;
			}

			if (ai_mesh->mAnimMeshes[i]->HasTangentsAndBitangents()) {
				PoolColorArray tangents;
				tangents.resize(num_vertices);
				PoolColorArray::Write w = tangents.write();
				for (int l = 0; l < num_vertices; l++) {
					_calc_tangent_from_mesh(ai_mesh, i, l, l, w);
				}
				PoolRealArray new_tangents = array_copy[VisualServer::ARRAY_TANGENT].duplicate(true);
				ERR_CONTINUE(new_tangents.size() != tangents.size() * 4);
				for (int l = 0; l < tangents.size(); l++) {
					new_tangents.write()[l + 0] = tangents[l].r;
					new_tangents.write()[l + 1] = tangents[l].g;
					new_tangents.write()[l + 2] = tangents[l].b;
					new_tangents.write()[l + 3] = tangents[l].a;
				}

				array_copy[VisualServer::ARRAY_TANGENT] = new_tangents;
			}

			morphs[i] = array_copy;
		}
		mesh->add_surface_from_arrays(primitive, array_mesh, morphs);
		mesh->surface_set_material(i, mat);
		mesh->surface_set_name(i, _ai_string_to_string(ai_mesh->mName));
		r_mesh_count++;
		print_line(String("Open Asset Importer: Created mesh (including instances) ") + _ai_string_to_string(ai_mesh->mName) + " " + itos(r_mesh_count) + " of " + itos(p_scene->mNumMeshes));
	}
	p_mesh_instance->set_mesh(mesh);
}
int EditorSceneImporterAlembic::index(Abc::ICompoundProperty iProp, Abc::PropertyHeader iHeader) {
	for (size_t i = 0; i < iProp.getNumProperties(); i++) {
		Abc::PropertyHeader header = iProp.getPropertyHeader(i);
		if (header.getName() == iHeader.getName()) {
			return i;
		}
	}
	return -1;
}

bool EditorSceneImporterAlembic::is_leaf(Abc::ICompoundProperty iProp, Abc::PropertyHeader iHeader) {
	if (!iProp.valid()) {
		return true;
	}

	int last = iProp.getNumProperties() - 1;
	Abc::PropertyHeader header = iProp.getPropertyHeader(last);
	if (header.getName() == iHeader.getName())
		return true;

	return false;
}

bool EditorSceneImporterAlembic::is_leaf(AbcG::IObject iObj) {
	if (!iObj.getParent().valid()) {
		return true;
	}

	Abc::IObject parent = iObj.getParent();
	int numChildren = parent.getNumChildren();

	Abc::IObject test = parent.getChild(numChildren - 1);
	if (test.valid() && test.getName() != iObj.getName()) {
		return false;
	}
	return true;
}
void EditorSceneImporterAlembic::tree(AbcG::IObject iObj, Node *p_root, Node *current, std::string prefix, real_t p_back_fps) {
	std::string path = iObj.getFullName();

	if (path == "/") {
		prefix = "";
	} else {
		if (iObj.getParent().getFullName() != "/") {
			prefix = prefix + "   ";
		}
		if (is_leaf(iObj)) {
			std::cout << prefix << " `--";
			prefix = prefix + " ";
		} else {
			std::cout << prefix << " |--";
			prefix = prefix + " |";
		}
	};
	std::cout << iObj.getName();
	current->set_name(iObj.getName().data());

	if (IXform::matches(iObj.getHeader())) {
		IXform xform(iObj.getParent(), iObj.getHeader().getName());
		IXformSchema &xs = xform.getSchema();
		XformSample xformSample;
		xs.get(xformSample);
		if (xs.getNumOps() > 0) {
			TimeSamplingPtr ts = xs.getTimeSampling();
			Abc::M44d abc_mat = xformSample.getMatrix();
			Transform3D ai_mat(
					(real_t)abc_mat[0][0],
					(real_t)abc_mat[0][1],
					(real_t)abc_mat[0][2],
					(real_t)abc_mat[0][3],
					(real_t)abc_mat[1][0],
					(real_t)abc_mat[1][1],
					(real_t)abc_mat[1][2],
					(real_t)abc_mat[1][3],
					(real_t)abc_mat[2][0],
					(real_t)abc_mat[2][1],
					(real_t)abc_mat[2][2],
					(real_t)abc_mat[2][3],
					(real_t)abc_mat[3][0],
					(real_t)abc_mat[3][1],
					(real_t)abc_mat[3][2],
					(real_t)abc_mat[3][3]);
			ai_mat.transpose();
			cast_to<Node3D>(current)->set_transform(ai_mat);
		}
	} else if (IPolyMesh::matches(iObj.getHeader())) {
		IPolyMesh polymesh(iObj.getParent(), iObj.getHeader().getName());
		std::string faceSetName;

		//TODO(Ernest) Multimesh?
		//TODO(Ernest) reserve number of meshes

		IPolyMeshSchema schema = polymesh.getSchema();
		IPolyMeshSchema::Sample mesh_samp;
		schema.get(mesh_samp);
		Ref<ArrayMesh> mesh = memnew(Ref<ArrayMesh>(ArrayMesh));
		if (faceSetName.length()) {
			mesh->set_name(faceSetName.data());
		} else {
			mesh->set_name(current->get_name());
		}
		Ref<SurfaceTool> surface_tool = memnew(Ref<SurfaceTool>(SurfaceTool));
		surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
		Array surface_arrays;
		surface_arrays.resize(Mesh::ARRAY_MAX);
		Vector<Vector3> vertices;
		Vector<Vector3> normals;
		Vector<Vector2> uvs;
		Vector<int32_t> faces;

		const size_t polyCount = mesh_samp.getFaceCounts()->size();
		{
			size_t begIndex = 0;
			const Imath::Vec3<float> *abc_positions = mesh_samp.getPositions()->get();
			for (size_t i = 0; i < polyCount; i++) {
				const int *face_indices = mesh_samp.getFaceIndices()->get();
				size_t faceCount = mesh_samp.getFaceCounts()->get()[i];
				if (faceCount > 2) {
					for (int j = faceCount - 1; j >= 0; --j) {
						int face_index = face_indices[begIndex + j];
						Vector3 pos;
						pos.x = abc_positions[face_index].x;
						pos.y = abc_positions[face_index].y;
						pos.z = abc_positions[face_index].z;
						vertices.push_back(pos);
					}
				}
			}

			if (schema.getNormalsParam().getNumSamples() > 1) {
				const IN3fGeomParam iNormals = schema.getNormalsParam();
				IN3fGeomParam::Sample normalSamp;
				if (iNormals) {
					iNormals.getExpanded(normalSamp);
				}
				const Imath::Vec3<float> *abc_normals = normalSamp.getVals()->get();
				for (int i = 0; i < polyCount; i++) {
					const int *face_indices = mesh_samp.getFaceIndices()->get();
					int faceCount = mesh_samp.getFaceCounts()->get()[i];
					if (faceCount > 2) {
						for (int j = 0; j < faceCount; j++) {
							int face_index = face_indices[begIndex + j];
							if (abc_normals) {
								Vector3 face_normal;
								face_normal.x = abc_normals[face_index].x;
								face_normal.y = abc_normals[face_index].y;
								face_normal.z = abc_normals[face_index].z;
								normals.push_back(face_normal);
							}
						}
					}
				}
			}

			surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
			surface_arrays[Mesh::ARRAY_NORMAL] = normals;
			surface_arrays[Mesh::ARRAY_TEX_UV] = uvs;
			surface_arrays[Mesh::ARRAY_INDEX] = faces;

			mesh = surface_tool->commit();
			MeshInstance3D mesh_instance = memnew(MeshInstance3D);
			mesh_instance->set_mesh(mesh);
			current->replace_by(mesh_instance);
		}

		std::vector<std::string> faceSetNames;
		schema.getFaceSetNames(faceSetNames);
		{
			std::vector<int32_t> facesetFaces;
			// TODO(Ernest) multi material
			for (size_t k = 0; k < faceSetNames.size(); k++) {
				if (schema.hasFaceSet(faceSetNames[k]) == false) {
					continue;
				}
				IFaceSetSchema faceSet = schema.getFaceSet(faceSetNames[k]).getSchema();
				IFaceSetSchema::Sample faceSetSamp;
				faceSet.get(faceSetSamp);
				const int *abcFacesetFaces = faceSetSamp.getFaces()->get();
				for (size_t j = 0; j < faceSetSamp.getFaces()->size(); j++) {
					facesetFaces.push_back(abcFacesetFaces[j]);
				}
			}

			if (schema.getUVsParam().getNumSamples() > 0) {
				const IV2fGeomParam iUVs = schema.getUVsParam();
				IV2fGeomParam::Sample uvSamp;
				if (iUVs) {
					iUVs.getExpanded(uvSamp);
				}
				const Imath::Vec2<float> *abc_uvs = uvSamp.getVals()->get();
				for (size_t l = 0; l < mesh->mNumVertices; l++) {
					aiVector2D ai_uv;
					ai_uv.x = abc_uvs[l].x;
					ai_uv.y = 1.0f - abc_uvs[l].y;
					uvs.push_back(ai_uv);
				}
				std::reverse(uvs.begin(), uvs.end());
				const size_t texture_coords = 1;
				for (size_t n = 0; n < texture_coords; ++n) {
					if (uvs.empty()) {
						break;
					}
					aiVector3D *out_uv = new aiVector3D[vertices.size()];
					for (size_t m = 0; m < uvs.size(); m++) {
						out_uv[m] = aiVector3D(uvs[m].x, uvs[m].y, 0.0f);
					}
					mesh->mTextureCoords[n] = out_uv;
					mesh->mNumUVComponents[n] = 2;
				}
			}
		}

		if (schema.getTopologyVariance() == kHomogenousTopology && schema.isConstant() == false) {
			TimeSamplingPtr ts = schema.getTimeSampling();
			size_t numChannels = schema.getNumSamples();
			for (size_t o = 0; o < numChannels; o++) {
				SampleTimeSet sampleTimes;
				MatrixSampleMap xformSamples;
				GetRelevantSampleTimes(o, 12.0, 0.0, 0.0, ts, numChannels, sampleTimes);

				for (SampleTimeSet::iterator I = sampleTimes.begin();
						I != sampleTimes.end(); ++I) {
					IPolyMeshSchema::Sample animMeshSamp;
					schema.get(animMeshSamp, Abc::ISampleSelector(*I));
					aiAnimMesh *animMesh = aiCreateAnimMesh(mesh);
					animMesh->mName = std::string("animation_") + std::to_string(animMeshes.size());
					const Imath::Vec3<float> *animPositions = animMeshSamp.getPositions()->get();
					size_t polyCount = animMeshSamp.getFaceCounts()->size();
					size_t begIndex = 0;
					std::vector<aiVector3D> animVertices;
					for (size_t p = 0; p < polyCount; p++) {
						const int *animFaceIndices = animMeshSamp.getFaceIndices()->get();
						int faceCount = animMeshSamp.getFaceCounts()->get()[p];
						if (faceCount > 2) {
							for (int j = faceCount - 1; j >= 0; --j) {
								int face_index = animFaceIndices[begIndex + j];
								aiVector3D pos;
								pos.x = animPositions[face_index].x;
								pos.y = animPositions[face_index].y;
								pos.z = animPositions[face_index].z;
								animVertices.push_back(pos);
							}
						}
						begIndex += faceCount;
					}
					animMesh->mNumVertices = static_cast<unsigned int>(animVertices.size());
					animMesh->mVertices = new aiVector3D[animVertices.size()];
					std::copy(animVertices.begin(), animVertices.end(), animMesh->mVertices);
					animMesh->mWeight = 1.0f;
					animMeshes.push_back(animMesh);
				}

				const aiMeshMorphAnim *anim_mesh = anim->mMorphMeshChannels[i];
				const String prop_name = _ai_string_to_string(anim_mesh->mName);
				const String mesh_name = prop_name.split("*")[0];
				if (p_removed_nodes.has(mesh_name)) {
					continue;
				}
				ERR_CONTINUE(prop_name.split("*").size() != 2);
				const MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(ap->get_owner()->find_node(mesh_name));
				ERR_CONTINUE(mesh_instance == NULL);
				if (ap->get_owner()->find_node(mesh_instance->get_name()) == NULL) {
					print_verbose("Can't find mesh in scene: " + mesh_instance->get_name());
					continue;
				}
				const String path = ap->get_owner()->get_path_to(mesh_instance);
				if (path.empty()) {
					print_verbose("Can't find mesh in scene");
					continue;
				}

				{
					size_t keys = 6;
					for (size_t q = 0; q < numChannels; q++) {
						aiMeshMorphAnim *meshMorphAnim = new aiMeshMorphAnim();
						aiString name = current->mName;
						name.Append("*");
						name.length = 1 + ASSIMP_itoa10(name.data + name.length, MAXLEN - 1, morphs.size());
						meshMorphAnim->mName.HashSet(name.C_Str());
						meshMorphAnim->mNumKeys = keys;
						meshMorphAnim->mKeys = new aiMeshMorphKey[keys];

						// Bracket the playing frame with weights of 0, during with 1 and after with 0.

						meshMorphAnim->mKeys[0].mNumValuesAndWeights = 1;
						meshMorphAnim->mKeys[0].mValues = new unsigned int[1];
						meshMorphAnim->mKeys[0].mWeights = new double[1];

						meshMorphAnim->mKeys[0].mValues[0] = q;
						meshMorphAnim->mKeys[0].mWeights[0] = 0.0f;
						meshMorphAnim->mKeys[0].mTime = 0.0f;

						meshMorphAnim->mKeys[1].mNumValuesAndWeights = 1;
						meshMorphAnim->mKeys[1].mValues = new unsigned int[1];
						meshMorphAnim->mKeys[1].mWeights = new double[1];

						meshMorphAnim->mKeys[1].mValues[0] = q;
						meshMorphAnim->mKeys[1].mWeights[0] = 0.0f;
						meshMorphAnim->mKeys[1].mTime = q - 0.01f;

						meshMorphAnim->mKeys[2].mNumValuesAndWeights = 1;
						meshMorphAnim->mKeys[2].mValues = new unsigned int[1];
						meshMorphAnim->mKeys[2].mWeights = new double[1];

						meshMorphAnim->mKeys[2].mValues[0] = q;
						meshMorphAnim->mKeys[2].mWeights[0] = 1.0f;
						meshMorphAnim->mKeys[2].mTime = q;

						meshMorphAnim->mKeys[3].mNumValuesAndWeights = 1;
						meshMorphAnim->mKeys[3].mValues = new unsigned int[1];
						meshMorphAnim->mKeys[3].mWeights = new double[1];

						meshMorphAnim->mKeys[3].mValues[0] = q;
						meshMorphAnim->mKeys[3].mWeights[0] = 1.0f;
						meshMorphAnim->mKeys[3].mTime = q + 1;

						meshMorphAnim->mKeys[4].mNumValuesAndWeights = 1;
						meshMorphAnim->mKeys[4].mValues = new unsigned int[1];
						meshMorphAnim->mKeys[4].mWeights = new double[1];

						meshMorphAnim->mKeys[4].mValues[0] = q;
						meshMorphAnim->mKeys[4].mWeights[0] = 0.0f;
						meshMorphAnim->mKeys[4].mTime = q + 1.01;

						meshMorphAnim->mKeys[5].mNumValuesAndWeights = 1;
						meshMorphAnim->mKeys[5].mValues = new unsigned int[1];
						meshMorphAnim->mKeys[5].mWeights = new double[1];

						meshMorphAnim->mKeys[5].mValues[0] = q;
						meshMorphAnim->mKeys[5].mWeights[0] = 0.0f;
						meshMorphAnim->mKeys[5].mTime = numChannels;

						##add to track
					}
				}

				ERR_CONTINUE(E == NULL);
				for (size_t k = 0; k < anim_mesh->mNumKeys; k++) {
					for (size_t j = 0; j < anim_mesh->mKeys[k].mNumValuesAndWeights; j++) {
						const HashMap<uint32_t, String>::Element *F = E->get().find(anim_mesh->mKeys[k].mValues[j]);
						ERR_CONTINUE(F == NULL);
						const String prop = "blend_shapes/" + F->get();
						const NodePath node_path = String(path) + ":" + prop;
						ERR_CONTINUE(ap->get_owner()->has_node(node_path) == false);
						int32_t blend_track_idx = -1;
						if (anim->find_track(node_path) == -1) {
							blend_track_idx = anim->get_track_count();
							anim->add_track(Animation::TYPE_VALUE);
							anim->track_set_interpolation_type(blend_track_idx, Animation::INTERPOLATION_LINEAR);
							anim->track_set_path(blend_track_idx, node_path);
						} else {
							blend_track_idx = anim->find_track(node_path);
						}
						float t = anim_mesh->mKeys[k].mTime / ticks_per_second;
						float w = anim_mesh->mKeys[k].mWeights[j];
						anim->track_insert_key(blend_track_idx, t, w);
					}
				}
			}
		}
	}

	if (iObj.getNumChildren()) {
		const unsigned int numChildren = iObj.getNumChildren();
		Node *nodes = memnew(Node);
		for (unsigned int i = 0; i < numChildren; i++) {
			Node *current_new_node = memnew(Node);
			nodes->add_child(current_new_node);
			current_new_node->set_owner(p_root);
		}
	}

	print("");

	// property tree
	Abc::ICompoundProperty props = iObj.getProperties();
	for (size_t i = 0; i < props.getNumProperties(); i++) {
		Abc::PropertyHeader header = props.getPropertyHeader(i);
		if (header.isScalar()) {
			tree(Abc::IScalarProperty(props, header.getName()), p_root, prefix);
		} else if (header.isArray()) {
			tree(Abc::IArrayProperty(props, header.getName()), p_root, prefix);
		} else {
			tree(Abc::ICompoundProperty(props, header.getName()), p_root, prefix);
		}
	}

	// object tree
	for (size_t i = 0; i < iObj.getNumChildren(); i++) {
		tree(AbcG::IObject(iObj, iObj.getChildHeader(i).getName()), p_root, current->get_child(i), showProps, prefix);
	};
}

void EditorSceneImporterAlembic::tree(Abc::ICompoundProperty iProp, Node *p_root, std::string prefix) {
	if (iProp.getObject().getFullName() != "/") {
		prefix = prefix + "   ";
	}
	if (is_leaf(iProp.getParent(), iProp.getHeader()) &&
			iProp.getObject().getNumChildren() == 0) {
		std::cout << prefix << " `--";
		prefix = prefix + " ";
	} else {
		if (is_leaf(iProp.getParent(), iProp.getHeader())) {
			std::cout << prefix << " | `--";
			prefix = prefix + " |";
		} else if (iProp.getObject().getNumChildren() == 0) {
			std::cout << prefix << " :--";
			prefix = prefix + " :";
		} else if (is_leaf(iProp, iProp.getHeader())) {
			std::cout << prefix << " | `--";
			prefix = prefix + " |";
		} else {
			std::cout << prefix << " | :--";
			prefix = prefix + " | :";
		}
	}

	std::cout << iProp.getName() << "\r" << std::endl;

	for (size_t i = 0; i < iProp.getNumProperties(); i++) {
		Abc::PropertyHeader header = iProp.getPropertyHeader(i);
		if (header.isScalar()) {
			tree(Abc::IScalarProperty(iProp, header.getName()), pScene, prefix);
		} else if (header.isArray()) {
			tree(Abc::IArrayProperty(iProp, header.getName()), pScene, prefix);
		} else {
			tree(Abc::ICompoundProperty(iProp, header.getName()), pScene, prefix);
		}
	}
}

void EditorSceneImporterAlembic::tree(Abc::IArrayProperty iProp, Node *p_root, std::string prefix) {
	if (iProp.getObject().getFullName() != "/") {
		prefix = prefix + "   ";
	}
	if (is_leaf(iProp.getParent(), iProp.getHeader()) &&
			(iProp.getObject().getNumChildren() == 0 ||
					iProp.getParent().getName() != "")) {
		std::cout << prefix << " `--";
	} else {
		std::cout << prefix << " :--";
		prefix = prefix + " :";
	}

	std::cout << iProp.getName() << "\r" << std::endl;
}

void EditorSceneImporterAlembic::GetRelevantSampleTimes(double frame, double fps, double shutterOpen, double shutterClose, AbcA::TimeSamplingPtr timeSampling, size_t numSamples, SampleTimeSet &output) {
	if (numSamples < 2) {
		output.insert(0.0);
		return;
	}

	chrono_t frameTime = frame / fps;

	chrono_t shutterOpenTime = (frame + shutterOpen) / fps;

	chrono_t shutterCloseTime = (frame + shutterClose) / fps;

	std::pair<index_t, chrono_t> shutterOpenFloor =
			timeSampling->getFloorIndex(shutterOpenTime, numSamples);

	std::pair<index_t, chrono_t> shutterCloseCeil =
			timeSampling->getCeilIndex(shutterCloseTime, numSamples);

	static const chrono_t epsilon = CMP_EPSILON;

	//check to see if our second sample is really the
	//floor that we want due to floating point slop
	//first make sure that we have at least two samples to work with
	if (shutterOpenFloor.first < shutterCloseCeil.first) {
		//if our open sample is less than open time,
		//look at the next index time
		if (shutterOpenFloor.second < shutterOpenTime) {
			chrono_t nextSampleTime =
					timeSampling->getSampleTime(shutterOpenFloor.first + 1);

			if (fabs(nextSampleTime - shutterOpenTime) < epsilon) {
				shutterOpenFloor.first += 1;
				shutterOpenFloor.second = nextSampleTime;
			}
		}
	}

	for (index_t i = shutterOpenFloor.first; i < shutterCloseCeil.first; ++i) {
		output.insert(timeSampling->getSampleTime(i));
	}

	//no samples above? put frame time in there and get out
	if (output.size() == 0) {
		output.insert(frameTime);
		return;
	}

	chrono_t lastSample = *(output.rbegin());

	//determine whether we need the extra sample at the end
	if ((fabs(lastSample - shutterCloseTime) > epsilon) && lastSample < shutterCloseTime) {
		output.insert(shutterCloseCeil.second);
	}
}

void EditorSceneImporterAlembic::tree(Abc::IScalarProperty iProp, Node *p_root, std::string prefix) {
	if (iProp.getObject().getFullName() != "/") {
		prefix = prefix + "   ";
	}
	if (is_leaf(iProp.getParent(), iProp.getHeader()) &&
			(iProp.getObject().getNumChildren() == 0 ||
					iProp.getParent().getName() != "")) {
		std::cout << prefix << " `--";
	} else {
		std::cout << prefix << " :--";
		prefix = prefix + " :";
	}

	std::cout << iProp.getName() << "\r" << std::endl;
}
