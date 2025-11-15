/**************************************************************************/
/*  usd_document.cpp                                                      */
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

#include "usd_document.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/math/color.h"
#include "scene/3d/bone_attachment_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"

#include "modules/gltf/extensions/gltf_light.h"
#include "modules/gltf/gltf_defines.h"
#include "modules/gltf/skin_tool.h"
#include "modules/gltf/structures/gltf_animation.h"
#include "modules/gltf/structures/gltf_camera.h"
#include "modules/gltf/structures/gltf_mesh.h"
#include "modules/gltf/structures/gltf_node.h"

#include <algorithm>
#include <vector>

#ifdef TOOLS_ENABLED
#include "editor/file_system/editor_file_system.h"
#endif

// FIXME: Hardcoded to avoid editor dependency.
#define USD_IMPORT_USE_NAMED_SKIN_BINDS 16
#define USD_IMPORT_DISCARD_MESHES_AND_MATERIALS 32
#define USD_IMPORT_FORCE_DISABLE_MESH_COMPRESSION 64

// TinyUSDZ is already included via usd_document.h

Transform3D USDDocument::_as_xform(const tinyusdz::value::matrix4d &p_mat) {
	Transform3D result;
	
	// TinyUSDZ matrix4d is a 4x4 matrix stored as array[16]
	// Format: column-major order
	// [0  4  8  12]
	// [1  5  9  13]
	// [2  6  10 14]
	// [3  7  11 15]
	
	// Extract translation (last column)
	result.origin = Vector3(
		real_t(p_mat.m[12]),
		real_t(p_mat.m[13]),
		real_t(p_mat.m[14])
	);
	
	// Extract rotation and scale from upper 3x3
	Basis basis;
	basis.set_column(0, Vector3(real_t(p_mat.m[0]), real_t(p_mat.m[1]), real_t(p_mat.m[2])));
	basis.set_column(1, Vector3(real_t(p_mat.m[4]), real_t(p_mat.m[5]), real_t(p_mat.m[6])));
	basis.set_column(2, Vector3(real_t(p_mat.m[8]), real_t(p_mat.m[9]), real_t(p_mat.m[10])));
	
	// Extract scale and normalize basis
	Vector3 scale = basis.get_scale();
	basis.orthonormalize();
	basis.scale(scale);
	result.basis = basis;
	
	return result;
}

Vector3 USDDocument::_as_vec3(const tinyusdz::value::float3 &p_vector) {
	return Vector3(real_t(p_vector[0]), real_t(p_vector[1]), real_t(p_vector[2]));
}

String USDDocument::_gen_unique_name(HashSet<String> &unique_names, const String &p_name) {
	const String s_name = p_name.validate_node_name();

	String u_name;
	int index = 1;
	while (true) {
		u_name = s_name;

		if (index > 1) {
			u_name += itos(index);
		}
		if (!unique_names.has(u_name)) {
			break;
		}
		index++;
	}

	unique_names.insert(u_name);

	return u_name;
}

String USDDocument::_sanitize_animation_name(const String &p_name) {
	String anim_name = p_name.validate_node_name();
	return AnimationLibrary::validate_library_name(anim_name);
}

String USDDocument::_gen_unique_animation_name(Ref<USDState> p_state, const String &p_name) {
	const String s_name = _sanitize_animation_name(p_name);

	String u_name;
	int index = 1;
	while (true) {
		u_name = s_name;

		if (index > 1) {
			u_name += itos(index);
		}
		if (!p_state->unique_animation_names.has(u_name)) {
			break;
		}
		index++;
	}

	p_state->unique_animation_names.insert(u_name);

	return u_name;
}

Error USDDocument::append_from_file(const String &p_path, Ref<GLTFState> p_state, uint32_t p_flags, const String &p_base_path) {
	Ref<USDState> state = p_state;
	ERR_FAIL_COND_V(state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_FILE_NOT_FOUND);
	if (p_state == Ref<USDState>()) {
		p_state.instantiate();
	}
	state->filename = p_path.get_file().get_basename();
	state->use_named_skin_binds = p_flags & USD_IMPORT_USE_NAMED_SKIN_BINDS;
	state->discard_meshes_and_materials = p_flags & USD_IMPORT_DISCARD_MESHES_AND_MATERIALS;
	
	String base_path = p_base_path;
	if (base_path.is_empty()) {
		base_path = p_path.get_base_dir();
	}
	state->base_path = base_path;
	
	Error err = _parse(state, p_path, base_path);
	ERR_FAIL_COND_V(err != OK, err);
	
	for (Ref<GLTFDocumentExtension> ext : document_extensions) {
		ERR_CONTINUE(ext.is_null());
		err = ext->import_post_parse(p_state);
		ERR_FAIL_COND_V(err != OK, err);
	}
	return OK;
}

Error USDDocument::append_from_buffer(const PackedByteArray &p_bytes, const String &p_base_path, Ref<GLTFState> p_state, uint32_t p_flags) {
	// TODO: Implement in-memory USD loading
	ERR_PRINT("USD: append_from_buffer not yet implemented");
	return ERR_UNAVAILABLE;
}

Error USDDocument::append_from_scene(Node *p_node, Ref<GLTFState> p_state, uint32_t p_flags) {
	// USD export functionality - not needed for import
	ERR_PRINT("USD: append_from_scene not yet implemented");
	return ERR_UNAVAILABLE;
}

Node *USDDocument::generate_scene(Ref<GLTFState> p_state, float p_bake_fps, bool p_trimming, bool p_remove_immutable_tracks) {
	// Use base class implementation
	return GLTFDocument::generate_scene(p_state, p_bake_fps, p_trimming, p_remove_immutable_tracks);
}

PackedByteArray USDDocument::generate_buffer(Ref<GLTFState> p_state) {
	// USD export functionality - not needed for import
	ERR_PRINT("USD: generate_buffer not yet implemented");
	return PackedByteArray();
}

Error USDDocument::write_to_filesystem(Ref<GLTFState> p_state, const String &p_path) {
	// USD export functionality - not needed for import
	ERR_PRINT("USD: write_to_filesystem not yet implemented");
	return ERR_UNAVAILABLE;
}

void USDDocument::set_naming_version(int p_version) {
	_naming_version = p_version;
}

int USDDocument::get_naming_version() const {
	return _naming_version;
}

// Export methods (merged from UsdDocument)
Error USDDocument::export_from_scene(Node *p_scene_root, Ref<USDState> p_state, int32_t p_flags) {
	// TODO: Update export functionality to use TinyUSDZ API
	// This method will convert the Godot scene to USD using TinyUSDZ
	ERR_PRINT("USD Export: export_from_scene not yet implemented with TinyUSDZ");
	return ERR_UNAVAILABLE;
}

String USDDocument::get_file_extension_for_format(bool p_binary) const {
	// Return the appropriate file extension based on the format
	return p_binary ? "usdc" : "usda";
}

Error USDDocument::_parse(Ref<USDState> p_state, const String &p_path, const String &p_base_path) {
	Error err;

	// Validate input
	if (p_path.is_empty()) {
		ERR_PRINT("USD: Empty file path provided");
		return ERR_FILE_NOT_FOUND;
	}

	// Check if file exists
	if (!FileAccess::exists(p_path)) {
		ERR_PRINT("USD: File does not exist: " + p_path);
		return ERR_FILE_NOT_FOUND;
	}

	// Load USD stage using TinyUSDZ
	tinyusdz::Stage stage;
	std::string warn;
	std::string err_msg;
	
	tinyusdz::USDLoadOptions options;
	options.do_composition = false; // Don't do composition for now
	
	bool ret = tinyusdz::LoadUSDFromFile(p_path.utf8().get_data(), &stage, &warn, &err_msg, options);
	
	if (!ret) {
		ERR_PRINT("USD: Failed to load USD file: " + p_path);
		if (!err_msg.empty()) {
			ERR_PRINT("USD: Error: " + String(err_msg.c_str()));
		}
		return ERR_FILE_CANT_OPEN;
	}
	
	if (!warn.empty()) {
		WARN_PRINT("USD: Warning: " + String(warn.c_str()));
	}
	
	// Validate stage
	if (stage.root_prims().empty()) {
		ERR_PRINT("USD: Stage has no root prims: " + p_path);
		return ERR_FILE_CORRUPT;
	}
	
	p_state->set_stage(stage);

	/* PARSE SCENE */
	err = _parse_scenes(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE NODES */
	err = _parse_nodes(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	if (!p_state->discard_meshes_and_materials) {
		/* PARSE IMAGES */
		err = _parse_images(p_state, p_base_path);
		ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

		/* PARSE MATERIALS */
		err = _parse_materials(p_state);
		ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);
	}

	/* PARSE SKINS */
	err = _parse_skins(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* DETERMINE SKELETONS */
	if (p_state->get_import_as_skeleton_bones()) {
		err = SkinTool::_determine_skeletons(p_state->skins, p_state->nodes, p_state->skeletons, p_state->root_nodes, true);
	} else {
		err = SkinTool::_determine_skeletons(p_state->skins, p_state->nodes, p_state->skeletons, Vector<GLTFNodeIndex>(), _naming_version < 2);
	}
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* CREATE SKELETONS */
	err = SkinTool::_create_skeletons(p_state->unique_names, p_state->skins, p_state->nodes, p_state->skeleton3d_to_usd_skeleton, p_state->skeletons, p_state->scene_nodes, _naming_version);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* CREATE SKINS */
	err = SkinTool::_create_skins(p_state->skins, p_state->nodes, p_state->use_named_skin_binds, p_state->unique_names);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE MESHES (we have enough info now) */
	err = _parse_meshes(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE LIGHTS */
	err = _parse_lights(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE CAMERAS */
	err = _parse_cameras(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* PARSE ANIMATIONS */
	err = _parse_animations(p_state);
	ERR_FAIL_COND_V(err != OK, ERR_PARSE_ERROR);

	/* ASSIGN SCENE NAMES */
	_assign_node_names(p_state);

	return OK;
}

Error USDDocument::_parse_scenes(Ref<USDState> p_state) {
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	// Get root prims
	const auto &root_prims = stage.root_prims();
	if (root_prims.empty()) {
		return ERR_INVALID_DATA;
	}

	// Use first root prim as scene name
	const tinyusdz::Prim &first_prim = root_prims[0];
	if (!first_prim.element_name().empty()) {
		p_state->scene_name = String(first_prim.element_name().c_str());
	}

	return OK;
}

// Helper function to recursively collect all prims from TinyUSDZ stage
static void _collect_prims_recursive(const tinyusdz::Prim &prim, const tinyusdz::Path &parent_path, 
		std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> &all_prims) {
	tinyusdz::Path prim_path = parent_path.AppendPrim(prim.element_name());
	all_prims.push_back({prim, prim_path});
	
	// Recursively process children
	for (const auto &child : prim.primChildren()) {
		_collect_prims_recursive(child, prim_path, all_prims);
	}
}

Error USDDocument::_parse_nodes(Ref<USDState> p_state) {
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	// Collect all prims recursively
	std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> all_prims;
	tinyusdz::Path root_path = tinyusdz::Path::make_root_path();
	
	for (const auto &root_prim : stage.root_prims()) {
		_collect_prims_recursive(root_prim, root_path, all_prims);
	}
	
	HashMap<String, GLTFNodeIndex> path_to_node_index;
	
	// First pass: create all nodes
	for (const auto &prim_path_pair : all_prims) {
		const tinyusdz::Prim &prim = prim_path_pair.first;
		const tinyusdz::Path &prim_path = prim_path_pair.second;
		
		Ref<GLTFNode> node;
		node.instantiate();

		// Set name
		String prim_name = String(prim.element_name().c_str());
		if (prim_name.is_empty()) {
			prim_name = "node_" + itos(p_state->nodes.size());
		}
		node->set_name(prim_name);
		node->set_original_name(prim_name);

		// Get transform from Xform if available
		if (prim.is<tinyusdz::Xform>()) {
			const tinyusdz::Xform *xform = prim.as<tinyusdz::Xform>();
			if (xform) {
				// TODO: Extract transform from Xform ops
				// For now, use identity transform
				node->transform = Transform3D();
			}
		} else {
			node->transform = Transform3D();
		}

		// Check for mesh
		if (prim.is<tinyusdz::GeomMesh>()) {
			// Will be set when parsing meshes
			node->mesh = -1; // Placeholder, will be set in _parse_meshes
		}

		// Check for camera
		if (prim.is<tinyusdz::GeomCamera>()) {
			// Will be set when parsing cameras
			node->camera = -1; // Placeholder, will be set in _parse_cameras
		}

		// Store node
		GLTFNodeIndex node_index = p_state->nodes.size();
		p_state->nodes.push_back(node);
		
		// Store prim path as string for lookup
		String prim_path_str = String(prim_path.prim_part().c_str());
		path_to_node_index[prim_path_str] = node_index;
		
		// Store prim path in node's additional_data for later matching
		node->set_additional_data("USD_prim_path", prim_path_str);
		
		// Cache in state for faster lookups
		p_state->prim_path_to_node_index[prim_path_str] = node_index;
	}

	// Second pass: build hierarchy
	for (const auto &prim_path_pair : all_prims) {
		const tinyusdz::Path &prim_path = prim_path_pair.second;
		String prim_path_str = String(prim_path.prim_part().c_str());
		
		if (!path_to_node_index.has(prim_path_str)) {
			continue;
		}
		
		GLTFNodeIndex node_index = path_to_node_index[prim_path_str];
		Ref<GLTFNode> node = p_state->nodes[node_index];

		// Set parent
		tinyusdz::Path parent_path = prim_path.GetParentPath();
		if (parent_path.is_valid() && !parent_path.is_root_path()) {
			String parent_path_str = String(parent_path.prim_part().c_str());
			if (path_to_node_index.has(parent_path_str)) {
				GLTFNodeIndex parent_index = path_to_node_index[parent_path_str];
				node->parent = parent_index;
				p_state->nodes[parent_index]->children.push_back(node_index);
			}
		} else {
			// Root node
			p_state->root_nodes.push_back(node_index);
		}
	}

	return OK;
}

// Helper to recursively find all mesh prims
static void _collect_mesh_prims_recursive(const tinyusdz::Prim &prim, const tinyusdz::Path &parent_path,
		std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> &mesh_prims) {
	tinyusdz::Path prim_path = parent_path.AppendPrim(prim.element_name());
	
	if (prim.is<tinyusdz::GeomMesh>()) {
		mesh_prims.push_back({prim, prim_path});
	}
	
	// Recursively process children
	for (const auto &child : prim.primChildren()) {
		_collect_mesh_prims_recursive(child, prim_path, mesh_prims);
	}
}

Error USDDocument::_parse_meshes(Ref<USDState> p_state) {
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	HashMap<String, GLTFMeshIndex> path_to_mesh_index;
	
	// Build material path to index map for faster lookup
	HashMap<String, GLTFMaterialIndex> material_name_to_index;
	for (GLTFMaterialIndex mat_i = 0; mat_i < p_state->materials.size(); mat_i++) {
		Ref<Material> mat = p_state->materials[mat_i];
		if (mat.is_valid()) {
			material_name_to_index[mat->get_name()] = mat_i;
		}
	}
	
	// Collect all mesh prims
	std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> mesh_prims;
	tinyusdz::Path root_path = tinyusdz::Path::make_root_path();
	
	for (const auto &root_prim : stage.root_prims()) {
		_collect_mesh_prims_recursive(root_prim, root_path, mesh_prims);
	}
	
	// Parse meshes
	GLTFMeshIndex mesh_index = 0;
	for (const auto &prim_path_pair : mesh_prims) {
		const tinyusdz::Prim &prim = prim_path_pair.first;
		const tinyusdz::Path &prim_path = prim_path_pair.second;
		
		if (!prim.is<tinyusdz::GeomMesh>()) {
			continue;
		}
		
		const tinyusdz::GeomMesh *usd_mesh = prim.as<tinyusdz::GeomMesh>();
		if (!usd_mesh) {
			continue;
		}

		Ref<GLTFMesh> gltf_mesh;
		gltf_mesh.instantiate();

		Ref<ImporterMesh> import_mesh;
		import_mesh.instantiate();
		String mesh_name = String(prim.element_name().c_str());
		if (mesh_name.is_empty()) {
			mesh_name = "mesh_" + itos(mesh_index);
		}
		import_mesh->set_name(_gen_unique_name(p_state->unique_mesh_names, mesh_name));

		// Get points (vertices)
		std::vector<tinyusdz::value::point3f> points = usd_mesh->get_points();
		if (points.empty()) {
			WARN_PRINT(vformat("USD: Mesh '%s' has no points, skipping", String(prim.element_name().c_str())));
			continue;
		}

		PackedVector3Array vertices;
		vertices.resize(points.size());
		for (size_t i = 0; i < points.size(); i++) {
			vertices.write[i] = _as_vec3(tinyusdz::value::float3(points[i][0], points[i][1], points[i][2]));
		}

		// Check for subdivision surfaces
		if (usd_mesh->subdivisionScheme.has_value()) {
			std::string scheme = usd_mesh->subdivisionScheme.value().str();
			if (scheme != "none" && !scheme.empty()) {
				WARN_PRINT(vformat("USD: Mesh '%s' uses subdivision scheme '%s'. Subdivision surfaces are not fully supported - importing control cage only.", 
					String(prim.element_name().c_str()), String(scheme.c_str())));
				// TODO: Implement subdivision surface tessellation
			}
		}

		// Get face vertex indices and counts
		std::vector<int32_t> face_vertex_indices = usd_mesh->get_faceVertexIndices();
		std::vector<int32_t> face_vertex_counts = usd_mesh->get_faceVertexCounts();
		
		if (!face_vertex_indices.empty() && !face_vertex_counts.empty()) {
				
				// Build index array for triangles
				PackedInt32Array indices;
				int index_offset = 0;
				for (size_t face_i = 0; face_i < face_vertex_counts.size(); face_i++) {
					int face_count = face_vertex_counts[face_i];
					if (face_count >= 3) {
						// Triangulate polygon
						for (int tri = 0; tri < face_count - 2; tri++) {
							indices.push_back(face_vertex_indices[index_offset]);
							indices.push_back(face_vertex_indices[index_offset + tri + 1]);
							indices.push_back(face_vertex_indices[index_offset + tri + 2]);
						}
					}
					index_offset += face_count;
				}

				// Get normals if available
				PackedVector3Array normals;
				std::vector<tinyusdz::value::normal3f> usd_normals = usd_mesh->get_normals();
				if (!usd_normals.empty()) {
					normals.resize(usd_normals.size());
					for (size_t i = 0; i < usd_normals.size(); i++) {
						normals.write[i] = _as_vec3(tinyusdz::value::float3(usd_normals[i][0], usd_normals[i][1], usd_normals[i][2]));
					}
				}

				// Get UVs - try standard "st" first, then check for other UV sets
				PackedVector2Array uvs;
				tinyusdz::GeomPrimvar uv_primvar;
				std::string err;
				
				// Try standard "st" primvar
				if (!usd_mesh->get_primvar("st", &uv_primvar, &err)) {
					// Try alternative names
					if (!usd_mesh->get_primvar("uv", &uv_primvar, &err)) {
						usd_mesh->get_primvar("map1", &uv_primvar, &err);
					}
				}
				
				if (uv_primvar.has_value()) {
					std::vector<tinyusdz::value::float2> usd_uvs;
					if (uv_primvar.get_value(&usd_uvs)) {
						// Handle different interpolation modes
						tinyusdz::Interpolation interp = tinyusdz::Interpolation::Vertex; // Default
						if (uv_primvar.has_interpolation()) {
							interp = uv_primvar.get_interpolation();
						}
						if (interp == tinyusdz::Interpolation::Vertex || interp == tinyusdz::Interpolation::Varying) {
							// Per-vertex UVs - map directly
							uvs.resize(usd_uvs.size());
							for (size_t i = 0; i < usd_uvs.size(); i++) {
								uvs.write[i] = Vector2(usd_uvs[i][0], 1.0f - usd_uvs[i][1]); // Flip V coordinate
							}
						} else if (interp == tinyusdz::Interpolation::FaceVarying) {
							// Per-face-vertex UVs - need to map through face vertex indices
							uvs.resize(face_vertex_indices.size());
							for (size_t i = 0; i < face_vertex_indices.size() && i < usd_uvs.size(); i++) {
								uvs.write[i] = Vector2(usd_uvs[i][0], 1.0f - usd_uvs[i][1]);
							}
						} else {
							// Constant or uniform - use first value for all
							if (usd_uvs.size() > 0) {
								Vector2 uv_val(usd_uvs[0][0], 1.0f - usd_uvs[0][1]);
								uvs.resize(vertices.size());
								for (size_t i = 0; i < uvs.size(); i++) {
									uvs.write[i] = uv_val;
								}
							}
						}
					}
				}

				// Get vertex colors (displayColor primvar)
				PackedColorArray colors;
				tinyusdz::GeomPrimvar color_primvar;
				if (usd_mesh->get_primvar("displayColor", &color_primvar, &err)) {
					std::vector<tinyusdz::value::color3f> usd_colors;
					if (color_primvar.get_value(&usd_colors)) {
						tinyusdz::Interpolation interp = tinyusdz::Interpolation::Vertex; // Default
						if (color_primvar.has_interpolation()) {
							interp = color_primvar.get_interpolation();
						}
						if (interp == tinyusdz::Interpolation::Vertex || interp == tinyusdz::Interpolation::Varying) {
							// Per-vertex colors
							colors.resize(usd_colors.size());
							for (size_t i = 0; i < usd_colors.size(); i++) {
								colors.write[i] = Color(usd_colors[i][0], usd_colors[i][1], usd_colors[i][2]);
							}
						} else if (interp == tinyusdz::Interpolation::FaceVarying) {
							// Per-face-vertex colors
							colors.resize(face_vertex_indices.size());
							for (size_t i = 0; i < face_vertex_indices.size() && i < usd_colors.size(); i++) {
								colors.write[i] = Color(usd_colors[i][0], usd_colors[i][1], usd_colors[i][2]);
							}
						} else {
							// Constant or uniform - use first value for all
							if (usd_colors.size() > 0) {
								Color color_val(usd_colors[0][0], usd_colors[0][1], usd_colors[0][2]);
								colors.resize(vertices.size());
								for (size_t i = 0; i < colors.size(); i++) {
									colors.write[i] = color_val;
								}
							}
						}
					}
				}

				// Build surface arrays
				Array surface_arrays;
				surface_arrays.resize(Mesh::ARRAY_MAX);
				surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
				if (indices.size() > 0) {
					surface_arrays[Mesh::ARRAY_INDEX] = indices;
				}
				if (normals.size() > 0) {
					surface_arrays[Mesh::ARRAY_NORMAL] = normals;
				}
				if (uvs.size() > 0) {
					surface_arrays[Mesh::ARRAY_TEX_UV] = uvs;
				}
				if (colors.size() > 0) {
					surface_arrays[Mesh::ARRAY_COLOR] = colors;
				}

				// TODO: Parse blend shapes if available
				// Design:
				// 1. Check if GeomMesh has blendShapeTargets relationship (GeomMesh::blendShapeTargets)
				// 2. Check if GeomMesh has blendShapes attribute (GeomMesh::blendShapes) - list of blend shape names
				// 3. For each blend shape target path:
				//    a. Find the BlendShape prim using stage.GetPrimAtPath() or recursive search
				//    b. Get BlendShape.offsets (vector3f[]) - vertex offsets
				//    c. Get BlendShape.normalOffsets (vector3f[]) - normal offsets
				//    d. Get BlendShape.pointIndices (int[]) - optional sparse indices
				// 4. Convert USD offset-based blend shapes to Godot absolute positions:
				//    - If pointIndices is empty: apply offsets[i] to base_vertices[i]
				//    - If pointIndices exists: apply offsets[i] to base_vertices[pointIndices[i]]
				// 5. Create Array for each blend shape with ARRAY_VERTEX and ARRAY_NORMAL
				// 6. Add to morphs array and call import_mesh->add_blend_shape(name)
				// Reference: tinyusdz::BlendShape struct in usdSkel.hh
				Array morphs;
				// TODO: Implement blend shape parsing
				if (false) { // Placeholder
					// Get blend shape targets
					UsdRelationship blend_shape_targets_rel = skel_binding_api.GetBlendShapeTargetsRel();
					VtTokenArray blend_shapes;
					skel_binding_api.GetBlendShapesAttr().Get(&blend_shapes);
					
					SdfPathVector blend_shape_targets;
					if (blend_shape_targets_rel.GetTargets(&blend_shape_targets) && !blend_shape_targets.empty()) {
						import_mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);
						
						// Add blend shape names
						for (size_t bs_i = 0; bs_i < blend_shapes.size() && bs_i < blend_shape_targets.size(); bs_i++) {
							String bs_name = String(blend_shapes[bs_i].GetText());
							if (bs_name.is_empty()) {
								bs_name = String("morph_") + itos(bs_i);
							}
							import_mesh->add_blend_shape(bs_name);
						}
						
						// Process each blend shape target
						for (size_t bs_i = 0; bs_i < blend_shape_targets.size(); bs_i++) {
							UsdSkelBlendShape blend_shape = UsdSkelBlendShape::Get(stage, blend_shape_targets[bs_i]);
							if (!blend_shape) {
								continue;
							}
							
							// Get offsets
							VtVec3fArray offsets;
							VtVec3fArray normal_offsets;
							VtIntArray point_indices;
							
							blend_shape.GetOffsetsAttr().Get(&offsets);
							blend_shape.GetNormalOffsetsAttr().Get(&normal_offsets);
							blend_shape.GetPointIndicesAttr().Get(&point_indices);
							
							// Create blend shape array
							Array array_copy;
							array_copy.resize(Mesh::ARRAY_MAX);
							
							// Copy base arrays
							for (int l = 0; l < Mesh::ARRAY_MAX; l++) {
								array_copy[l] = surface_arrays[l];
							}
							
							// Apply offsets to vertices (USD stores offsets, Godot needs absolute positions)
							if (!offsets.empty()) {
								PackedVector3Array varr = surface_arrays[Mesh::ARRAY_VERTEX];
								const Vector<Vector3> src_varr = varr;
								
								// Resize to match base mesh
								varr.resize(src_varr.size());
								
								if (point_indices.empty()) {
									// Apply to all points - USD offsets are per-point in order
									for (size_t i = 0; i < offsets.size() && i < varr.size(); i++) {
										varr.write[i] = src_varr[i] + _as_vec3(offsets[i]);
									}
									// Fill remaining with base values
									for (size_t i = offsets.size(); i < varr.size(); i++) {
										varr.write[i] = src_varr[i];
									}
								} else {
									// Initialize with base values
									for (size_t i = 0; i < varr.size(); i++) {
										varr.write[i] = src_varr[i];
									}
									// Apply to specific point indices
									for (size_t i = 0; i < point_indices.size() && i < offsets.size(); i++) {
										int idx = point_indices[i];
										if (idx >= 0 && idx < varr.size()) {
											varr.write[idx] = src_varr[idx] + _as_vec3(offsets[i]);
										}
									}
								}
								array_copy[Mesh::ARRAY_VERTEX] = varr;
							} else {
								// No offsets, use base vertices
								array_copy[Mesh::ARRAY_VERTEX] = surface_arrays[Mesh::ARRAY_VERTEX];
							}
							
							// Apply normal offsets
							if (!normal_offsets.empty()) {
								PackedVector3Array narr;
								if (surface_arrays[Mesh::ARRAY_NORMAL].get_type() == Variant::PACKED_VECTOR3_ARRAY) {
									narr = surface_arrays[Mesh::ARRAY_NORMAL];
								} else {
									// Create empty normals array
									narr.resize(vertices.size());
								}
								
								if (narr.size() > 0) {
									const Vector<Vector3> src_narr = narr;
									
									// Resize to match base mesh
									narr.resize(src_narr.size());
									
									if (point_indices.empty()) {
										// Apply to all normals
										for (size_t i = 0; i < normal_offsets.size() && i < narr.size(); i++) {
											narr.write[i] = src_narr[i] + _as_vec3(normal_offsets[i]);
										}
										// Fill remaining with base values
										for (size_t i = normal_offsets.size(); i < narr.size(); i++) {
											narr.write[i] = src_narr[i];
										}
									} else {
										// Initialize with base values
										for (size_t i = 0; i < narr.size(); i++) {
											narr.write[i] = src_narr[i];
										}
										// Apply to specific point indices
										for (size_t i = 0; i < point_indices.size() && i < normal_offsets.size(); i++) {
											int idx = point_indices[i];
											if (idx >= 0 && idx < narr.size()) {
												narr.write[idx] = src_narr[idx] + _as_vec3(normal_offsets[i]);
											}
										}
									}
									array_copy[Mesh::ARRAY_NORMAL] = narr;
								}
							} else {
								// No normal offsets, use base normals
								if (surface_arrays[Mesh::ARRAY_NORMAL].get_type() == Variant::PACKED_VECTOR3_ARRAY) {
									array_copy[Mesh::ARRAY_NORMAL] = surface_arrays[Mesh::ARRAY_NORMAL];
								}
							}
							
							// Enforce blend shape mask array format (only vertex and normal)
							for (int l = 0; l < Mesh::ARRAY_MAX; l++) {
								if (!(Mesh::ARRAY_FORMAT_BLEND_SHAPE_MASK & (1ULL << l))) {
									array_copy[l] = Variant();
								}
							}
							
							morphs.push_back(array_copy);
						}
					}
				}
				
				import_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, surface_arrays, morphs, Dictionary(), Ref<Material>());
			} else {
				WARN_PRINT(vformat("USD: Mesh '%s' has invalid face data, skipping", String(prim.element_name().c_str())));
				continue;
			}
		} else {
			WARN_PRINT(vformat("USD: Mesh '%s' has no face data, creating point cloud", String(prim.element_name().c_str())));
			// Create a point cloud if we have vertices but no faces
			Array surface_arrays;
			surface_arrays.resize(Mesh::ARRAY_MAX);
			surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
			import_mesh->add_surface(Mesh::PRIMITIVE_POINTS, surface_arrays, Array(), Dictionary(), Ref<Material>());
		}

		if (import_mesh->get_surface_count() == 0) {
			WARN_PRINT(vformat("USD: Mesh '%s' has no surfaces, skipping", String(prim.element_name().c_str())));
			continue;
		}

		// TODO: Get material binding and assign to mesh
		// Design:
		// 1. Check GPrim::materialBinding (Relationship) - this is the primary binding
		// 2. Also check GPrim::materialBindingPreview and GPrim::materialBindingFull for purpose-specific bindings
		// 3. Follow the relationship target path to find the Material prim:
		//    - Use stage.GetPrimAtPath() or search recursively
		//    - Verify it's a Material prim using prim.is<tinyusdz::Material>()
		// 4. Once we have the Material, look it up in material_path_to_index map
		// 5. Assign material to mesh using gltf_mesh->set_instance_materials()
		// 6. For multi-surface meshes, may need to check GeomSubset prims for per-face material assignments
		// Reference: GPrim::MaterialBinding mixin in usdGeom.hh, Material struct in usdShade.hh
		// For now, skip material binding - will be implemented later
		
		gltf_mesh->set_mesh(import_mesh);
		p_state->meshes.push_back(gltf_mesh);
		String prim_path_str = String(prim_path.prim_part().c_str());
		path_to_mesh_index[prim_path_str] = mesh_index;

		// Find corresponding node and set mesh index using cached mapping
		
		if (p_state->prim_path_to_node_index.has(prim_path_str)) {
			GLTFNodeIndex node_index = p_state->prim_path_to_node_index[prim_path_str];
			if (node_index >= 0 && node_index < p_state->nodes.size()) {
				p_state->nodes[node_index]->mesh = mesh_index;
			}
		}

		mesh_index++;
	}

	print_verbose("USD: Total meshes: " + itos(p_state->meshes.size()));

	return OK;
}

// Helper to recursively find all material prims
static void _collect_material_prims_recursive(const tinyusdz::Prim &prim, const tinyusdz::Path &parent_path,
		std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> &material_prims) {
	tinyusdz::Path prim_path = parent_path.AppendPrim(prim.element_name());
	
	if (prim.is<tinyusdz::Material>()) {
		material_prims.push_back({prim, prim_path});
	}
	
	// Recursively process children
	for (const auto &child : prim.primChildren()) {
		_collect_material_prims_recursive(child, prim_path, material_prims);
	}
}

Error USDDocument::_parse_materials(Ref<USDState> p_state) {
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	HashMap<String, GLTFMaterialIndex> material_path_to_index;
	
	// Collect all material prims
	std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> material_prims;
	tinyusdz::Path root_path = tinyusdz::Path::make_root_path();
	
	for (const auto &root_prim : stage.root_prims()) {
		_collect_material_prims_recursive(root_prim, root_path, material_prims);
	}
	
	// Parse materials
	for (const auto &prim_path_pair : material_prims) {
		const tinyusdz::Prim &prim = prim_path_pair.first;
		const tinyusdz::Path &prim_path = prim_path_pair.second;
		
		if (!prim.is<tinyusdz::Material>()) {
			continue;
		}
		
		const tinyusdz::Material *material = prim.as<tinyusdz::Material>();
		if (!material) {
			continue;
		}
		
		Ref<StandardMaterial3D> godot_material;
		godot_material.instantiate();
		godot_material->set_name(String(prim.element_name().c_str()));
		godot_material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		
		GLTFMaterialIndex material_index = p_state->materials.size();
		String prim_path_str = String(prim_path.prim_part().c_str());
		material_path_to_index[prim_path_str] = material_index;
		
		// Get the surface output connection
		// TODO: Follow surface connection to find shader
		// For now, create a basic material
		// TinyUSDZ's Material has a `surface` TypedConnection that points to a shader
		// We need to follow this connection to get the UsdPreviewSurface shader
		
		// Placeholder: create default material
		godot_material->set_albedo(Color(0.8, 0.8, 0.8));
		godot_material->set_metallic(0.0);
		godot_material->set_roughness(0.5);
		
		// TODO: Parse surface connection and UsdPreviewSurface shader inputs
		// Design:
		// 1. Get Material::surface (TypedConnection<value::token>) - this connects to a shader
		// 2. Follow connection to find shader prim:
		//    - material.surface.get_target_path() gives the path to the shader
		//    - Use stage.GetPrimAtPath() or search recursively to find the prim
		//    - Check if prim.is<tinyusdz::Shader>() and get shader.info_id
		// 3. If info_id == "UsdPreviewSurface", cast to UsdPreviewSurface:
		//    - Check if prim.value (value::Value) contains UsdPreviewSurface
		//    - Use tydra::GetPreviewSurface() or manually extract from prim.value
		// 4. Parse UsdPreviewSurface inputs:
		//    - diffuseColor (color3f) -> set_albedo()
		//    - metallic (float) -> set_metallic()
		//    - roughness (float) -> set_roughness()
		//    - emissiveColor (color3f) -> set_emission()
		//    - opacity (float) -> set_transparency() if < 1.0
		//    - normal (normal3f) -> may be connected to texture
		// 5. For texture connections (e.g., diffuseColor connected to UsdUVTexture):
		//    - Follow TypedConnection to find UsdUVTexture shader
		//    - Get UsdUVTexture::file (AssetPath) attribute
		//    - Resolve asset path and find/create texture in p_state->textures
		//    - Assign texture to material using set_texture()
		// 6. Handle UsdPrimvarReader nodes for UV coordinates and other primvars
		// Reference: Material, Shader, UsdPreviewSurface, UsdUVTexture in usdShade.hh
		// Note: TinyUSDZ uses value::Value type-erased storage, may need tydra API for evaluation
		
		p_state->materials.push_back(godot_material);
		material_path_to_index[prim_path_str] = p_state->materials.size() - 1;
	}

	print_verbose("USD: Total materials: " + itos(p_state->materials.size()));

	return OK;
}

// Helper to recursively find all skeleton prims
static void _collect_skeleton_prims_recursive(const tinyusdz::Prim &prim, const tinyusdz::Path &parent_path,
		std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> &skeleton_prims) {
	tinyusdz::Path prim_path = parent_path.AppendPrim(prim.element_name());
	
	if (prim.is<tinyusdz::Skeleton>()) {
		skeleton_prims.push_back({prim, prim_path});
	}
	
	// Recursively process children
	for (const auto &child : prim.primChildren()) {
		_collect_skeleton_prims_recursive(child, prim_path, skeleton_prims);
	}
}

Error USDDocument::_parse_skins(Ref<USDState> p_state) {
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	HashMap<GLTFNodeIndex, bool> joint_mapping;
	HashMap<String, GLTFSkeletonIndex> skeleton_path_to_index;
	
	// Collect all skeleton prims
	std::vector<std::pair<tinyusdz::Prim, tinyusdz::Path>> skeleton_prims;
	tinyusdz::Path root_path = tinyusdz::Path::make_root_path();
	
	for (const auto &root_prim : stage.root_prims()) {
		_collect_skeleton_prims_recursive(root_prim, root_path, skeleton_prims);
	}
	
	// First pass: find all skeletons and create skeleton nodes
	for (const auto &prim_path_pair : skeleton_prims) {
		const tinyusdz::Prim &prim = prim_path_pair.first;
		const tinyusdz::Path &prim_path = prim_path_pair.second;
		
		if (!prim.is<tinyusdz::Skeleton>()) {
			continue;
		}
		
		const tinyusdz::Skeleton *usd_skeleton = prim.as<tinyusdz::Skeleton>();
		if (!usd_skeleton) {
			continue;
		}
		
		GLTFSkeletonIndex skeleton_index = p_state->skeletons.size();
		String prim_path_str = String(prim_path.prim_part().c_str());
		skeleton_path_to_index[prim_path_str] = skeleton_index;
		
		Ref<GLTFSkeleton> skeleton;
		skeleton.instantiate();
		skeleton->set_name(String(prim.element_name().c_str()));
		
		// TODO: Get joints - Need to evaluate TypedAttribute
		// Design:
		// 1. Use tydra::EvaluateTypedAttribute() or manually evaluate:
		//    - Check if usd_skeleton->joints.has_value()
		//    - Get default value: joints.get_default_value<std::vector<value::token>>()
		//    - Or evaluate at specific time if timesampled
		// 2. For each joint token, resolve to full path:
		//    - Joint tokens are relative to skeleton prim
		//    - Build full path: prim_path.AppendPrim(joint_token.str())
		//    - Or use prim_path + "/" + joint_token.str()
		// 3. Find corresponding GLTFNode by matching USD_prim_path
		// 4. Add to skin->joints and mark node as joint
		// Reference: TypedAttribute evaluation in tinyusdz, Skeleton struct in usdSkel.hh
		std::vector<tinyusdz::value::token> joints;
		if (usd_skeleton->joints.has_value()) {
			// TODO: Evaluate attribute - see design above
		}
		
		// TODO: Get bind transforms - Need to evaluate TypedAttribute
		// Design: Similar to joints, evaluate bindTransforms attribute
		// Convert each matrix4d to Transform3D using _as_xform()
		// Store inverse bind matrices in skin->inverse_binds
		std::vector<tinyusdz::value::matrix4d> bind_transforms;
		if (usd_skeleton->bindTransforms.has_value()) {
			// TODO: Evaluate attribute - see design above
		}
		
		// TODO: Get rest transforms (if available) - Need to evaluate TypedAttribute
		// Design: Similar to bind transforms, evaluate restTransforms attribute
		// Store in node's additional_data as "GODOT_rest_transform"
		std::vector<tinyusdz::value::matrix4d> rest_transforms;
		if (usd_skeleton->restTransforms.has_value()) {
			// TODO: Evaluate attribute - see design above
		}
			
		// Create a skin for this skeleton
		Ref<GLTFSkin> skin;
		skin.instantiate();
		skin->set_name(String(prim.element_name().c_str()) + "_skin");
		
		// Process joints
		// TODO: Implement full joint processing once we can evaluate TypedAttributes
		// For now, create empty skeleton/skin
		// This will need to:
		// 1. Evaluate joints attribute to get joint paths
		// 2. Find corresponding nodes by path
		// 3. Evaluate bindTransforms and restTransforms
		// 4. Create skin with proper inverse bind matrices
		
		if (skin->joints.size() > 0) {
			p_state->skins.push_back(skin);
			skeleton->set_skin(p_state->skins.size() - 1);
		}
		
		p_state->skeletons.push_back(skeleton);
	}
	
	// TODO: Second pass: find skinned meshes and bind them to skeletons
	// Design:
	// 1. Iterate through all GeomMesh prims (reuse mesh collection from _parse_meshes)
	// 2. Check GeomMesh::skeleton (Relationship) - points to Skeleton prim
	// 3. Follow relationship to get skeleton path, look up in skeleton_path_to_index
	// 4. Get joint influences from primvars:
	//    - Get primvar "skel:jointIndices" (int[]) - joint indices per vertex
	//    - Get primvar "skel:jointWeights" (float[]) - joint weights per vertex
	//    - Use GeomMesh::get_primvar("skel:jointIndices") and get_primvar("skel:jointWeights")
	// 5. Store joint indices/weights in mesh's additional_data or create GLTFSkin structure
	// 6. Assign skin index to mesh node using p_state->nodes[node_index]->skin
	// 7. For GLTF compatibility, may need to convert to GLTF skin format
	// Reference: GeomMesh::skeleton, GeomPrimvar API in usdGeom.hh
	
	p_state->original_skin_indices = p_state->skin_indices.duplicate();
	
	// Use SkinTool to process skins
	if (p_state->skins.size() > 0) {
		Error err = SkinTool::_asset_parse_skins(
				p_state->original_skin_indices,
				p_state->skins.duplicate(),
				p_state->nodes.duplicate(),
				p_state->skin_indices,
				p_state->skins,
				joint_mapping);
		if (err != OK) {
			return err;
		}
		
		for (int i = 0; i < p_state->skins.size(); ++i) {
			Ref<GLTFSkin> skin = p_state->skins.write[i];
			ERR_FAIL_COND_V(skin.is_null(), ERR_PARSE_ERROR);
			// Expand and verify the skin
			ERR_FAIL_COND_V(SkinTool::_expand_skin(p_state->nodes, skin), ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(SkinTool::_verify_skin(p_state->nodes, skin), ERR_PARSE_ERROR);
		}
	}

	print_verbose("USD: Total skins: " + itos(p_state->skins.size()));
	print_verbose("USD: Total skeletons: " + itos(p_state->skeletons.size()));

	return OK;
}

Error USDDocument::_parse_animations(Ref<USDState> p_state) {
	// TODO: Implement animation parsing using TinyUSDZ API
	// Design:
	// 1. Find SkelAnimation prims (prim.is<tinyusdz::SkelAnimation>())
	// 2. Evaluate SkelAnimation attributes:
	//    - translations (vector3f[]) - per-joint translations over time
	//    - rotations (quatf[]) - per-joint rotations over time
	//    - scales (vector3f[]) - per-joint scales over time
	//    - joints (token[]) - joint names this animation affects
	// 3. Handle time samples:
	//    - Check if attributes are timesampled using has_timesamples()
	//    - Get all time codes from time samples
	//    - Evaluate at each time code to get keyframe values
	// 4. Match joints to GLTF nodes by path/name
	// 5. Create GLTFAnimation with node tracks:
	//    - For each joint, create translation/rotation/scale tracks
	//    - Add keyframes at each time sample
	// 6. Also handle Xform animation (non-skeletal):
	//    - Find Xform prims with animated xformOps
	//    - Evaluate xformOps at different time codes
	//    - Create animation tracks for transform changes
	// 7. Handle blend shape animations:
	//    - Check SkelAnimation::blendShapes (token[]) - blend shape names
	//    - Get blend shape weights over time
	//    - Create blend shape animation tracks
	// Reference: SkelAnimation struct in usdSkel.hh, Xform ops in xform.hh
	// Note: TinyUSDZ uses TypedTimeSamples for animated attributes
	
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	// TODO: Get stage time range from stage metadata
	// Check StageMetas::startTimeCode and endTimeCode
	double start_time = 0.0;
	double end_time = 1.0;
	
	// TODO: Collect all SkelAnimation prims recursively
	HashMap<String, GLTFAnimationIndex> animation_path_to_index;
	
	// TODO: Implement animation parsing - see design above
	// All the code below uses OpenUSD APIs and needs to be rewritten
	if (false) { // Placeholder - remove when implementing
		// Old OpenUSD code removed - see design above for implementation
	}
	
	return OK; // Placeholder
}

Error USDDocument::_parse_cameras(Ref<USDState> p_state) {
	// TODO: Implement camera parsing using TinyUSDZ API
	// Design:
	// 1. Find GeomCamera prims (prim.is<tinyusdz::GeomCamera>())
	// 2. Evaluate GeomCamera attributes:
	//    - focalLength (float) - camera focal length
	//    - horizontalAperture (float) - horizontal film aperture
	//    - verticalAperture (float) - vertical film aperture
	//    - clippingRange (float2) - near and far clipping planes
	//    - projection (token) - "perspective" or "orthographic"
	// 3. Convert USD camera to GLTF camera:
	//    - For perspective: calculate FOV from focalLength and aperture
	//    - FOV = 2 * atan(aperture / (2 * focalLength))
	//    - For orthographic: use xmag/ymag from aperture
	// 4. Create GLTFCamera and add to p_state->cameras
	// 5. Find corresponding node and set camera index
	// Reference: GeomCamera struct in usdGeom.hh
	
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	HashMap<String, GLTFCameraIndex> camera_path_to_index;
	
	// TODO: Collect all GeomCamera prims recursively
	// TODO: Implement camera parsing - see design above
	
	print_verbose("USD: Total cameras: " + itos(p_state->cameras.size()));

	return OK;
}

Error USDDocument::_parse_lights(Ref<USDState> p_state) {
	// TODO: Implement light parsing using TinyUSDZ API
	// Design:
	// 1. Find light prims - TinyUSDZ has various light types in usdLux.hh:
	//    - SphereLight, DiskLight, RectLight, CylinderLight, etc.
	//    - All inherit from Light base class
	// 2. Check prim.is<tinyusdz::SphereLight>() or other light types
	// 3. Evaluate common light attributes (from Light base class):
	//    - color (color3f) - light color
	//    - intensity (float) - light intensity
	//    - exposure (float) - exposure value
	//    - enableColorTemperature (bool) - use color temperature
	//    - colorTemperature (float) - color temperature in Kelvin
	// 4. Evaluate type-specific attributes:
	//    - SphereLight: radius, treatPointAsDirection
	//    - DiskLight: radius
	//    - RectLight: width, height
	//    - DistantLight: angle (for directional lights)
	//    - SpotLight: innerConeAngle, outerConeAngle
	// 5. Convert to GLTFLight:
	//    - Map USD light types to GLTF light types
	//    - Calculate final color: color * intensity * pow(2, exposure)
	//    - For spot lights, set inner/outer cone angles
	// 6. Create GLTFLight and add to p_state->lights
	// 7. Find corresponding node and set light index
	// Reference: Light types in usdLux.hh (SphereLight, DiskLight, etc.)
	
	const tinyusdz::Stage &stage = p_state->get_stage();
	
	// TODO: Collect all light prims recursively
	// USD uses UsdLux for lights
	// TODO: Implement light parsing - see design above
	
	print_verbose("USD: Total lights: " + itos(p_state->lights.size()));

	return OK;
}

Error USDDocument::_parse_images(Ref<USDState> p_state, const String &p_base_path) {
	// TODO: Implement image/texture parsing using TinyUSDZ API
	// Design:
	// 1. Find all Material prims and follow shader connections
	// 2. For each UsdPreviewSurface shader, check texture inputs:
	//    - diffuseColor, emissiveColor, normal, metallicRoughness, occlusion
	// 3. Follow TypedConnection to find UsdUVTexture shaders
	// 4. Get UsdUVTexture::file (AssetPath) attribute
	// 5. Resolve asset paths (may need AssetResolutionResolver)
	// 6. Load images and create GLTFTexture entries
	// 7. Store texture paths for later material assignment
	// Reference: Material, Shader, UsdUVTexture in usdShade.hh
	// Note: Asset path resolution may require TinyUSDZ's AssetResolutionResolver
	
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	const tinyusdz::Stage &stage = p_state->get_stage();

	HashSet<String> texture_paths;
	
	// TODO: Collect texture paths from materials - see design above
	// For now, return OK (no textures parsed)
	
	print_verbose("USD: Total textures: " + itos(p_state->textures.size()));

	return OK;
}
												if (resolved_path.is_empty()) {
													resolved_path = String(asset_path.GetAssetPath().c_str());
												}
												if (!resolved_path.is_empty()) {
													texture_paths.insert(resolved_path);
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// Load images from collected paths
	for (const String &texture_path : texture_paths) {
		String path = texture_path;
		// Use only filename for absolute paths to avoid portability issues.
		if (path.is_absolute_path()) {
			path = path.get_file();
		}
		if (!p_base_path.is_empty()) {
			path = p_base_path.path_join(path);
		}
		path = path.simplify_path();

		// Try to load as existing texture first
		String base_dir = p_state->get_base_path();
		Ref<Texture2D> texture = ResourceLoader::load(_get_texture_path(base_dir, path), "Texture2D");
		if (texture.is_valid()) {
			p_state->images.push_back(texture);
			p_state->source_images.push_back(texture->get_image());
			continue;
		}

		// Fallback to loading as byte array
		Vector<uint8_t> data = FileAccess::get_file_as_bytes(path);
		if (data.is_empty()) {
			WARN_PRINT(vformat("USD: Image couldn't be loaded from path: %s. Skipping it.", path));
			p_state->images.push_back(Ref<Texture2D>()); // Placeholder to keep count.
			p_state->source_images.push_back(Ref<Image>());
			continue;
		}

		// Parse the image data from bytes into an Image resource and save if needed.
		String file_extension = path.get_extension();
		Ref<Image> img = _parse_image_bytes_into_image(p_state, data, path, p_state->images.size());
		img->set_name(path.get_file().get_basename());
		GLTFImageIndex image_index = _parse_image_save_image(p_state, data, file_extension, p_state->images.size(), img);
		if (image_index >= 0) {
			// Image was added successfully
		}
	}

	// Create a texture for each image
	for (GLTFImageIndex i = 0; i < p_state->images.size(); i++) {
		Ref<GLTFTexture> texture;
		texture.instantiate();
		texture->set_src_image(i);
		p_state->textures.push_back(texture);
	}

	print_verbose("USD: Total images: " + itos(p_state->images.size()));

	return OK;
}

String USDDocument::_get_texture_path(const String &p_base_directory, const String &p_source_file_path) const {
	// Check if the original path exists first.
	if (FileAccess::exists(p_source_file_path)) {
		return p_source_file_path.strip_edges();
	}
	const String tex_file_name = p_source_file_path.get_file();
	const Vector<String> subdirs = {
		"", "textures/", "Textures/", "images/",
		"Images/", "materials/", "Materials/",
		"maps/", "Maps/", "tex/", "Tex/"
	};
	String base_dir = p_base_directory;
	const String source_file_name = tex_file_name;
	while (!base_dir.is_empty()) {
		String old_base_dir = base_dir;
		for (int i = 0; i < subdirs.size(); ++i) {
			String full_path = base_dir.path_join(subdirs[i] + source_file_name);
			if (FileAccess::exists(full_path)) {
				return full_path.strip_edges();
			}
		}
		base_dir = base_dir.get_base_dir();
		if (base_dir == old_base_dir) {
			break;
		}
	}
	return String();
}

Ref<Texture2D> USDDocument::_get_texture(Ref<USDState> p_state, const GLTFTextureIndex p_texture, int p_texture_type) {
	ERR_FAIL_INDEX_V(p_texture, p_state->textures.size(), Ref<Texture2D>());
	Ref<GLTFTexture> gltf_texture = p_state->textures[p_texture];
	ERR_FAIL_COND_V(gltf_texture.is_null(), Ref<Texture2D>());
	GLTFImageIndex image_index = gltf_texture->get_src_image();
	ERR_FAIL_INDEX_V(image_index, p_state->images.size(), Ref<Texture2D>());
	return p_state->images[image_index];
}

Ref<Image> USDDocument::_parse_image_bytes_into_image(Ref<USDState> p_state, const Vector<uint8_t> &p_bytes, const String &p_filename, int p_index) {
	Ref<Image> r_image;
	r_image.instantiate();
	// Try to import first based on filename.
	String filename_lower = p_filename.to_lower();
	if (filename_lower.ends_with(".png")) {
		r_image->load_png_from_buffer(p_bytes);
	} else if (filename_lower.ends_with(".jpg") || filename_lower.ends_with(".jpeg")) {
		r_image->load_jpg_from_buffer(p_bytes);
	} else if (filename_lower.ends_with(".tga")) {
		r_image->load_tga_from_buffer(p_bytes);
	} else if (filename_lower.ends_with(".webp")) {
		r_image->load_webp_from_buffer(p_bytes);
	}
	// If we didn't pass the above tests, try loading as each option.
	if (r_image->is_empty()) { // Try PNG first.
		r_image->load_png_from_buffer(p_bytes);
	}
	if (r_image->is_empty()) { // And then JPEG.
		r_image->load_jpg_from_buffer(p_bytes);
	}
	if (r_image->is_empty()) { // And then TGA.
		r_image->load_tga_from_buffer(p_bytes);
	}
	// If it still can't be loaded, give up and insert an empty image as placeholder.
	if (r_image->is_empty()) {
		ERR_PRINT(vformat("USD: Couldn't load image index '%d'", p_index));
	}
	return r_image;
}

GLTFImageIndex USDDocument::_parse_image_save_image(Ref<USDState> p_state, const Vector<uint8_t> &p_bytes, const String &p_file_extension, int p_index, Ref<Image> p_image) {
	if (p_image.is_null() || p_image->is_empty()) {
		return -1;
	}

#ifdef TOOLS_ENABLED
	GLTFState::HandleBinaryImageMode handling = p_state->handle_binary_image_mode;
	if (handling == GLTFState::HANDLE_BINARY_IMAGE_MODE_DISCARD_ALL_TEXTURES) {
		return -1;
	}

	if (handling == GLTFState::HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES) {
		String file_path = p_state->get_base_path().path_join(".tmp_textures").path_join(itos(p_index) + "_" + p_image->get_name() + "." + p_file_extension);
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		da->make_dir_recursive(file_path.get_base_dir());
		Error err = OK;
		if (p_file_extension.is_empty()) {
			// If a file extension was not specified, save the image data to a PNG file.
			err = p_image->save_png(file_path);
			ERR_FAIL_COND_V(err != OK, -1);
		} else {
			// If a file extension was specified, save the original bytes to a file with that extension.
			Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE, &err);
			ERR_FAIL_COND_V(err != OK, -1);
			file->store_buffer(p_bytes);
			file->close();
		}
		HashMap<StringName, Variant> custom_options;
		custom_options[SNAME("mipmaps/generate")] = true;
		EditorFileSystem::get_singleton()->update_file(file_path);
		EditorFileSystem::get_singleton()->reimport_append(file_path, custom_options, String(), Dictionary());
		Ref<Texture2D> saved_image = ResourceLoader::load(_get_texture_path(p_state->get_base_path(), file_path), "Texture2D");
		if (saved_image.is_valid()) {
			p_state->images.push_back(saved_image);
			p_state->source_images.push_back(saved_image->get_image());
			return p_state->images.size() - 1;
		}
	}
#endif // TOOLS_ENABLED

	// Fallback: create texture directly from image
	Ref<ImageTexture> texture;
	texture.instantiate();
	texture->set_image(p_image);
	p_state->images.push_back(texture);
	p_state->source_images.push_back(p_image);
	return p_state->images.size() - 1;
}

BoneAttachment3D *USDDocument::_generate_bone_attachment(Ref<USDState> p_state, Skeleton3D *p_skeleton, const GLTFNodeIndex p_node_index, const GLTFNodeIndex p_bone_index) {
	// TODO: Implement bone attachment generation
	// This is called by GLTFDocument::generate_scene() to create BoneAttachment3D nodes
	// Should create a BoneAttachment3D node and attach it to the skeleton
	// Reference: GLTFDocument::_generate_scene_node() implementation
	return nullptr;
}

ImporterMeshInstance3D *USDDocument::_generate_mesh_instance(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement mesh instance generation
	// This is called by GLTFDocument::generate_scene() to create ImporterMeshInstance3D nodes
	// Should create an ImporterMeshInstance3D node with the mesh from p_state->nodes[p_node_index]->mesh
	// Reference: GLTFDocument::_generate_scene_node() implementation
	return nullptr;
}

Camera3D *USDDocument::_generate_camera(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement camera generation
	// This is called by GLTFDocument::generate_scene() to create Camera3D nodes
	// Should create a Camera3D node with settings from p_state->cameras[p_state->nodes[p_node_index]->camera]
	// Reference: GLTFDocument::_generate_scene_node() implementation
	return nullptr;
}

Light3D *USDDocument::_generate_light(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement light generation
	// This is called by GLTFDocument::generate_scene() to create Light3D nodes
	// Should create appropriate Light3D subclass (DirectionalLight3D, OmniLight3D, SpotLight3D)
	// based on p_state->lights[p_state->nodes[p_node_index]->light]
	// Reference: GLTFDocument::_generate_scene_node() implementation
	return nullptr;
}

Node3D *USDDocument::_generate_spatial(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement spatial node generation
	// This is called by GLTFDocument::generate_scene() to create generic Node3D nodes
	// Should create a Node3D node with transform from p_state->nodes[p_node_index]->transform
	// Reference: GLTFDocument::_generate_scene_node() implementation
	return nullptr;
}

void USDDocument::_assign_node_names(Ref<USDState> p_state) {
	// Use base class implementation
	GLTFDocument::_assign_node_names(p_state);
}

void USDDocument::_process_mesh_instances(Ref<USDState> p_state, Node *p_scene_root) {
	// Use base class implementation
	GLTFDocument::_process_mesh_instances(p_state, p_scene_root);
}

void USDDocument::_generate_scene_node(Ref<USDState> p_state, const GLTFNodeIndex p_node_index, Node *p_scene_parent, Node *p_scene_root) {
	// Use base class implementation
	GLTFDocument::_generate_scene_node(p_state, p_node_index, p_scene_parent, p_scene_root);
}

void USDDocument::_generate_skeleton_bone_node(Ref<USDState> p_state, const GLTFNodeIndex p_node_index, Node *p_scene_parent, Node *p_scene_root) {
	// Use base class implementation
	GLTFDocument::_generate_skeleton_bone_node(p_state, p_node_index, p_scene_parent, p_scene_root);
}

void USDDocument::_import_animation(Ref<USDState> p_state, AnimationPlayer *p_animation_player, const GLTFAnimationIndex p_index, const bool p_trimming, const bool p_remove_immutable_tracks) {
	// Use base class implementation
	GLTFDocument::_import_animation(p_state, p_animation_player, p_index, p_trimming, p_remove_immutable_tracks);
}

