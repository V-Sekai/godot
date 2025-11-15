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
#include "usd_import_state.h"
#include "usd_mesh_export_helper.h"

// Include Texture2D here to avoid conflict with TinyUSDZ's Texture struct in header
#include "scene/resources/texture.h"

// TinyUSDZ writer headers for export
#include "usda-writer.hh"
#include "usdc-writer.hh"

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
	
	// TinyUSDZ matrix4d is a 4x4 matrix stored as m[4][4] (row-major order)
	// Format: row-major order
	// [m[0][0] m[0][1] m[0][2] m[0][3]]
	// [m[1][0] m[1][1] m[1][2] m[1][3]]
	// [m[2][0] m[2][1] m[2][2] m[2][3]]
	// [m[3][0] m[3][1] m[3][2] m[3][3]]
	// Translation is at m[3][0], m[3][1], m[3][2]
	
	// Extract translation (last row, first 3 columns)
	result.origin = Vector3(
		real_t(p_mat.m[3][0]),
		real_t(p_mat.m[3][1]),
		real_t(p_mat.m[3][2])
	);
	
	// Extract rotation and scale from upper 3x3 (row-major, so columns are m[row][col])
	Basis basis;
	basis.set_column(0, Vector3(real_t(p_mat.m[0][0]), real_t(p_mat.m[1][0]), real_t(p_mat.m[2][0])));
	basis.set_column(1, Vector3(real_t(p_mat.m[0][1]), real_t(p_mat.m[1][1]), real_t(p_mat.m[2][1])));
	basis.set_column(2, Vector3(real_t(p_mat.m[0][2]), real_t(p_mat.m[1][2]), real_t(p_mat.m[2][2])));
	
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
	// Cast to USDState
	Ref<USDState> usd_state = p_state;
	if (usd_state.is_null()) {
		ERR_PRINT("USD Export: Invalid state");
		return ERR_INVALID_PARAMETER;
	}
	
	const tinyusdz::Stage &stage = usd_state->get_stage();
	
	// Determine format from file extension
	bool is_binary = p_path.ends_with(".usdc") || p_path.ends_with(".usdz");
	
	std::string warn;
	std::string err;
	bool ret = false;
	
	if (is_binary) {
		// Use USDC writer for binary format
		ret = tinyusdz::usdc::SaveAsUSDCToFile(p_path.utf8().get_data(), stage, &warn, &err);
	} else {
		// Use USDA writer for ASCII format
		ret = tinyusdz::usda::SaveAsUSDA(p_path.utf8().get_data(), stage, &warn, &err);
	}
	
	if (!ret) {
		ERR_PRINT("USD Export: Failed to save file: " + p_path);
		if (!err.empty()) {
			ERR_PRINT("USD Export: Error: " + String(err.c_str()));
		}
		return ERR_FILE_CANT_WRITE;
	}
	
	if (!warn.empty()) {
		WARN_PRINT("USD Export: Warning: " + String(warn.c_str()));
	}
	
	return OK;
}

void USDDocument::set_naming_version(int p_version) {
	_naming_version = p_version;
}

int USDDocument::get_naming_version() const {
	return _naming_version;
}

// Export methods (merged from UsdDocument)
Error USDDocument::export_from_scene(Node *p_scene_root, Ref<USDState> p_state, int32_t p_flags) {
	if (!p_scene_root) {
		ERR_PRINT("USD Export: Invalid scene root");
		return ERR_INVALID_PARAMETER;
	}

	// Create a new TinyUSDZ stage
	tinyusdz::Stage stage;
	
	// Minimum working subset: export first MeshInstance3D found in scene
	// TODO: Full scene traversal and hierarchy export
	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(p_scene_root);
	if (!mesh_instance) {
		// Try to find a MeshInstance3D in children
		for (int i = 0; i < p_scene_root->get_child_count(); i++) {
			Node *child = p_scene_root->get_child(i);
			mesh_instance = Object::cast_to<MeshInstance3D>(child);
			if (mesh_instance) {
				break;
			}
		}
	}
	
	if (mesh_instance && mesh_instance->get_mesh().is_valid()) {
		Ref<Mesh> mesh = mesh_instance->get_mesh();
		String mesh_name = mesh_instance->get_name();
		if (mesh_name.is_empty()) {
			mesh_name = "mesh";
		}
		String mesh_path_str = "/" + mesh_name;
		tinyusdz::Path mesh_path(mesh_path_str.utf8().get_data(), "");
		
		UsdMeshExportHelper helper;
		tinyusdz::Prim mesh_prim = helper.export_geom_mesh(mesh, &stage, mesh_path);
		
		if (!stage.add_root_prim(std::move(mesh_prim))) {
			ERR_PRINT("USD Export: Failed to add mesh prim to stage: " + String(stage.get_error().c_str()));
			return ERR_CANT_CREATE;
		}
	} else {
		// Create empty root prim if no mesh found
		tinyusdz::Xform root_xform;
		root_xform.name = "root";
		tinyusdz::Prim root_prim(root_xform);
		if (!stage.add_root_prim(std::move(root_prim))) {
			ERR_PRINT("USD Export: Failed to add root prim to stage: " + String(stage.get_error().c_str()));
			return ERR_CANT_CREATE;
		}
	}
	
	// Commit stage
	stage.commit();
	
	// Store stage in state
	p_state->set_stage(stage);
	
	return OK;
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
	for (const auto &child : prim.children()) {
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
				node->set_xform(Transform3D());
			}
		} else {
			node->set_xform(Transform3D());
		}

		// Check for mesh
		if (prim.is<tinyusdz::GeomMesh>()) {
			// Will be set when parsing meshes
			node->set_mesh(-1); // Placeholder, will be set in _parse_meshes
		}

		// Check for camera
		if (prim.is<tinyusdz::GeomCamera>()) {
			// Will be set when parsing cameras
			node->set_camera(-1); // Placeholder, will be set in _parse_cameras
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
		tinyusdz::Path parent_path = prim_path.get_parent_path();
		if (parent_path.is_valid() && !parent_path.is_root_path()) {
			String parent_path_str = String(parent_path.prim_part().c_str());
			if (path_to_node_index.has(parent_path_str)) {
				GLTFNodeIndex parent_index = path_to_node_index[parent_path_str];
				node->set_parent(parent_index);
				Vector<int> parent_children = p_state->nodes[parent_index]->get_children();
				parent_children.push_back(node_index);
				p_state->nodes[parent_index]->set_children(parent_children);
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
	for (const auto &child : prim.children()) {
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
			tinyusdz::value::float3 vec = {{points[i][0], points[i][1], points[i][2]}};
			vertices.write[i] = _as_vec3(vec);
		}

		// Check for subdivision surfaces
		// subdivisionScheme is a TypedAttributeWithFallback<SubdivisionScheme>
		// Get the value directly (it has a fallback, so get_value() always returns a value)
		tinyusdz::GeomMesh::SubdivisionScheme scheme = usd_mesh->subdivisionScheme.get_value();
		if (scheme != tinyusdz::GeomMesh::SubdivisionScheme::SubdivisionSchemeNone) {
			// Convert enum to string for warning message
			String scheme_str = "unknown";
			switch (scheme) {
				case tinyusdz::GeomMesh::SubdivisionScheme::CatmullClark:
					scheme_str = "catmullClark";
					break;
				case tinyusdz::GeomMesh::SubdivisionScheme::Loop:
					scheme_str = "loop";
					break;
				case tinyusdz::GeomMesh::SubdivisionScheme::Bilinear:
					scheme_str = "bilinear";
					break;
				case tinyusdz::GeomMesh::SubdivisionScheme::SubdivisionSchemeNone:
					scheme_str = "none";
					break;
			}
			WARN_PRINT(vformat("USD: Mesh '%s' uses subdivision scheme '%s'. Subdivision surfaces are not fully supported - importing control cage only.", 
				String(prim.element_name().c_str()), scheme_str));
			// TODO: Implement subdivision surface tessellation
		}

		// Get face vertex indices and counts
		std::vector<int32_t> face_vertex_indices = usd_mesh->get_faceVertexIndices();
		std::vector<int32_t> face_vertex_counts = usd_mesh->get_faceVertexCounts();
		
		// Simplified mesh import: only vertices, faces, and basic normals
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

			// Get normals if available (simplified - just per-vertex)
			PackedVector3Array normals;
			std::vector<tinyusdz::value::normal3f> usd_normals = usd_mesh->get_normals();
			if (!usd_normals.empty() && usd_normals.size() == vertices.size()) {
				normals.resize(usd_normals.size());
				for (size_t i = 0; i < usd_normals.size(); i++) {
					tinyusdz::value::float3 vec = {{usd_normals[i][0], usd_normals[i][1], usd_normals[i][2]}};
					normals.write[i] = _as_vec3(vec);
				}
			}

			// Build surface arrays (minimal: vertices, indices, normals only)
			Array surface_arrays;
			surface_arrays.resize(Mesh::ARRAY_MAX);
			surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
			if (indices.size() > 0) {
				surface_arrays[Mesh::ARRAY_INDEX] = indices;
			}
			if (normals.size() > 0) {
				surface_arrays[Mesh::ARRAY_NORMAL] = normals;
			}
			
			// Skip UVs, colors, blend shapes for minimum working subset
			Array morphs;
			import_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, surface_arrays, morphs, Dictionary(), Ref<Material>());
		} else if (!vertices.is_empty()) {
			// Create a point cloud if we have vertices but no faces
			WARN_PRINT(vformat("USD: Mesh '%s' has no face data, creating point cloud", String(prim.element_name().c_str())));
			Array surface_arrays;
			surface_arrays.resize(Mesh::ARRAY_MAX);
			surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
			import_mesh->add_surface(Mesh::PRIMITIVE_POINTS, surface_arrays, Array(), Dictionary(), Ref<Material>());
		} else {
			WARN_PRINT(vformat("USD: Mesh '%s' has no valid geometry, skipping", String(prim.element_name().c_str())));
			continue;
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
				p_state->nodes[node_index]->set_mesh(mesh_index);
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
	for (const auto &child : prim.children()) {
		_collect_material_prims_recursive(child, prim_path, material_prims);
	}
}

Error USDDocument::_parse_materials(Ref<USDState> p_state) {
	// Stubbed out for minimum working subset - skip material parsing
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
	for (const auto &child : prim.children()) {
		_collect_skeleton_prims_recursive(child, prim_path, skeleton_prims);
	}
}

Error USDDocument::_parse_skins(Ref<USDState> p_state) {
	// Stubbed out for minimum working subset - skip skeleton/skin parsing
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

