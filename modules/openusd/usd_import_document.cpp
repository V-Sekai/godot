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

#include "usd_import_document.h"

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

// USD headers
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/base/vt/types.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformOp.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/bindingAPI.h>

PXR_NAMESPACE_USING_DIRECTIVE

Transform3D USDDocument::_as_xform(const GfMatrix4d &p_mat) {
	Transform3D result;
	
	// Extract translation
	GfVec3d translation = p_mat.ExtractTranslation();
	result.origin = Vector3(translation[0], translation[1], translation[2]);
	
	// Extract rotation and scale
	GfMatrix3d rot_scale = p_mat.ExtractRotationMatrix();
	GfVec3d scale = p_mat.ExtractScale();
	
	// Build basis from rotation matrix and scale
	Basis basis;
	basis.set_column(0, Vector3(rot_scale[0][0] / scale[0], rot_scale[0][1] / scale[0], rot_scale[0][2] / scale[0]));
	basis.set_column(1, Vector3(rot_scale[1][0] / scale[1], rot_scale[1][1] / scale[1], rot_scale[1][2] / scale[1]));
	basis.set_column(2, Vector3(rot_scale[2][0] / scale[2], rot_scale[2][1] / scale[2], rot_scale[2][2] / scale[2]));
	basis.orthonormalize();
	
	// Apply scale
	basis.scale(Vector3(scale[0], scale[1], scale[2]));
	result.basis = basis;
	
	return result;
}

Vector3 USDDocument::_as_vec3(const GfVec3f &p_vector) {
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

	// Open USD stage
	UsdStageRefPtr stage = UsdStage::Open(p_path.utf8().get_data());
	if (!stage) {
		ERR_PRINT("USD: Failed to open USD file: " + p_path);
		ERR_PRINT("USD: This may be due to an invalid USD file or missing USD dependencies");
		return ERR_FILE_CANT_OPEN;
	}
	
	// Validate stage
	if (!stage->GetRootLayer()) {
		ERR_PRINT("USD: Stage has no root layer: " + p_path);
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
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		return ERR_INVALID_DATA;
	}

	// Get default prim or root prim
	UsdPrim default_prim = stage->GetDefaultPrim();
	if (!default_prim.IsValid()) {
		// Try to get root prims
		UsdPrimRange range = stage->Traverse();
		if (range.empty()) {
			return ERR_INVALID_DATA;
		}
		default_prim = *range.begin();
	}

	if (default_prim.IsValid()) {
		p_state->scene_name = String(default_prim.GetName().GetText());
	}

	return OK;
}

Error USDDocument::_parse_nodes(Ref<USDState> p_state) {
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		return ERR_INVALID_DATA;
	}

	HashMap<SdfPath, GLTFNodeIndex> path_to_node_index;
	UsdPrimRange range = stage->Traverse();
	
	// First pass: create all nodes
	for (UsdPrim prim : range) {
		Ref<GLTFNode> node;
		node.instantiate();

		// Set name
		String prim_name = String(prim.GetName().GetText());
		if (prim_name.is_empty()) {
			prim_name = String(prim.GetPath().GetName().GetText());
		}
		node->set_name(prim_name);
		node->set_original_name(prim_name);

		// Get transform
		UsdGeomXformable xformable(prim);
		if (xformable) {
			GfMatrix4d local_transform;
			bool reset_xform_stack = false;
			xformable.GetLocalTransformation(&local_transform, &reset_xform_stack);
			node->transform = _as_xform(local_transform);
		}

		// Check for mesh
		UsdGeomMesh mesh(prim);
		if (mesh) {
			// Will be set when parsing meshes
			node->mesh = -1; // Placeholder, will be set in _parse_meshes
		}

		// Check for camera
		UsdGeomCamera camera(prim);
		if (camera) {
			// Will be set when parsing cameras
			node->camera = -1; // Placeholder, will be set in _parse_cameras
		}

		// Store node
		GLTFNodeIndex node_index = p_state->nodes.size();
		p_state->nodes.push_back(node);
		SdfPath prim_path = prim.GetPath();
		path_to_node_index[prim_path] = node_index;
		
		// Store prim path in node's additional_data for later matching
		String prim_path_str = String(prim_path.GetText());
		node->set_additional_data("USD_prim_path", prim_path_str);
		
		// Cache in state for faster lookups
		p_state->prim_path_to_node_index[prim_path_str] = node_index;
	}

	// Second pass: build hierarchy
	for (UsdPrim prim : range) {
		SdfPath prim_path = prim.GetPath();
		if (!path_to_node_index.has(prim_path)) {
			continue;
		}
		
		GLTFNodeIndex node_index = path_to_node_index[prim_path];
		Ref<GLTFNode> node = p_state->nodes[node_index];

		// Set parent
		UsdPrim parent_prim = prim.GetParent();
		if (parent_prim && parent_prim != stage->GetPseudoRoot()) {
			SdfPath parent_path = parent_prim.GetPath();
			if (path_to_node_index.has(parent_path)) {
				GLTFNodeIndex parent_index = path_to_node_index[parent_path];
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

Error USDDocument::_parse_meshes(Ref<USDState> p_state) {
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		ERR_PRINT("USD: Invalid stage in _parse_meshes");
		return ERR_INVALID_DATA;
	}

	HashMap<SdfPath, GLTFMeshIndex> path_to_mesh_index;
	
	// Build material path to index map for faster lookup
	HashMap<String, GLTFMaterialIndex> material_name_to_index;
	for (GLTFMaterialIndex mat_i = 0; mat_i < p_state->materials.size(); mat_i++) {
		Ref<Material> mat = p_state->materials[mat_i];
		if (mat.is_valid()) {
			material_name_to_index[mat->get_name()] = mat_i;
		}
	}
	
	// Parse meshes
	GLTFMeshIndex mesh_index = 0;
	UsdPrimRange range = stage->Traverse();
	for (UsdPrim prim : range) {
		UsdGeomMesh usd_mesh(prim);
		if (!usd_mesh) {
			continue;
		}

		Ref<GLTFMesh> gltf_mesh;
		gltf_mesh.instantiate();

		Ref<ImporterMesh> import_mesh;
		import_mesh.instantiate();
		String mesh_name = String(prim.GetName().GetText());
		if (mesh_name.is_empty()) {
			mesh_name = "mesh_" + itos(mesh_index);
		}
		import_mesh->set_name(_gen_unique_name(p_state->unique_mesh_names, mesh_name));

		// Get points (vertices)
		UsdAttribute points_attr = usd_mesh.GetPointsAttr();
		VtArray<GfVec3f> points;
		if (!points_attr || !points_attr.Get(&points) || points.empty()) {
			WARN_PRINT(vformat("USD: Mesh '%s' has no points, skipping", String(prim.GetName().GetText())));
			continue;
		}

		PackedVector3Array vertices;
		vertices.resize(points.size());
		for (size_t i = 0; i < points.size(); i++) {
			vertices.write[i] = _as_vec3(points[i]);
		}

		// Check for subdivision surfaces
		TfToken subdivision_scheme;
		usd_mesh.GetSubdivisionSchemeAttr().Get(&subdivision_scheme);
		if (subdivision_scheme != UsdGeomTokens->none && subdivision_scheme != TfToken()) {
			WARN_PRINT(vformat("USD: Mesh '%s' uses subdivision scheme '%s'. Subdivision surfaces are not fully supported - importing control cage only.", 
				String(prim.GetName().GetText()), String(subdivision_scheme.GetText())));
			// TODO: Implement subdivision surface tessellation
			// For now, we'll import the control cage as-is
		}

		// Get face vertex indices
		UsdAttribute face_vertex_indices_attr = usd_mesh.GetFaceVertexIndicesAttr();
		UsdAttribute face_vertex_counts_attr = usd_mesh.GetFaceVertexCountsAttr();
		
		VtArray<int> face_vertex_indices;
		VtArray<int> face_vertex_counts;
		
		if (face_vertex_indices_attr && face_vertex_indices_attr.Get(&face_vertex_indices) &&
			face_vertex_counts_attr && face_vertex_counts_attr.Get(&face_vertex_counts) &&
			!face_vertex_indices.empty() && !face_vertex_counts.empty()) {
				
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
				UsdAttribute normals_attr = usd_mesh.GetNormalsAttr();
				VtArray<GfVec3f> usd_normals;
				if (normals_attr && normals_attr.Get(&usd_normals)) {
					normals.resize(usd_normals.size());
					for (size_t i = 0; i < usd_normals.size(); i++) {
						normals.write[i] = _as_vec3(usd_normals[i]);
					}
				}

				// Get UVs - try standard "st" first, then check for other UV sets
				PackedVector2Array uvs;
				UsdGeomPrimvarsAPI primvars_api(usd_mesh);
				
				// Try standard "st" primvar
				UsdGeomPrimvar uv_primvar = usd_mesh.GetPrimvar(TfToken("st"));
				if (!uv_primvar) {
					// Try alternative names
					uv_primvar = usd_mesh.GetPrimvar(TfToken("uv"));
					if (!uv_primvar) {
						uv_primvar = usd_mesh.GetPrimvar(TfToken("map1"));
					}
				}
				
				if (uv_primvar) {
					VtArray<GfVec2f> usd_uvs;
					if (uv_primvar.Get(&usd_uvs)) {
						// Handle different interpolation modes
						TfToken interp = uv_primvar.GetInterpolation();
						if (interp == UsdGeomTokens->vertex || interp == UsdGeomTokens->varying) {
							// Per-vertex UVs - map directly
							uvs.resize(usd_uvs.size());
							for (size_t i = 0; i < usd_uvs.size(); i++) {
								uvs.write[i] = Vector2(usd_uvs[i][0], 1.0f - usd_uvs[i][1]); // Flip V coordinate
							}
						} else if (interp == UsdGeomTokens->faceVarying) {
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
				UsdGeomPrimvar color_primvar = usd_mesh.GetPrimvar(TfToken("displayColor"));
				if (color_primvar) {
					VtArray<GfVec3f> usd_colors;
					if (color_primvar.Get(&usd_colors)) {
						TfToken interp = color_primvar.GetInterpolation();
						if (interp == UsdGeomTokens->vertex || interp == UsdGeomTokens->varying) {
							// Per-vertex colors
							colors.resize(usd_colors.size());
							for (size_t i = 0; i < usd_colors.size(); i++) {
								colors.write[i] = Color(usd_colors[i][0], usd_colors[i][1], usd_colors[i][2]);
							}
						} else if (interp == UsdGeomTokens->faceVarying) {
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

				// Parse blend shapes if available
				Array morphs;
				UsdSkelBindingAPI skel_binding_api(prim);
				if (skel_binding_api) {
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
				WARN_PRINT(vformat("USD: Mesh '%s' has invalid face data, skipping", String(prim.GetName().GetText())));
				continue;
			}
		} else {
			WARN_PRINT(vformat("USD: Mesh '%s' has no face data, creating point cloud", String(prim.GetName().GetText())));
			// Create a point cloud if we have vertices but no faces
			Array surface_arrays;
			surface_arrays.resize(Mesh::ARRAY_MAX);
			surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
			import_mesh->add_surface(Mesh::PRIMITIVE_POINTS, surface_arrays, Array(), Dictionary(), Ref<Material>());
		}

		if (import_mesh->get_surface_count() == 0) {
			WARN_PRINT(vformat("USD: Mesh '%s' has no surfaces, skipping", String(prim.GetName().GetText())));
			continue;
		}

		// Get material binding and assign to mesh
		UsdShadeMaterialBindingAPI binding_api(prim);
		UsdShadeMaterial bound_material = binding_api.ComputeBoundMaterial();
		if (!bound_material) {
			// Try legacy binding
			UsdRelationship mat_rel = prim.GetRelationship(TfToken("material:binding"));
			if (mat_rel) {
				SdfPathVector targets;
				if (mat_rel.GetTargets(&targets) && !targets.empty()) {
					bound_material = UsdShadeMaterial::Get(stage, targets[0]);
				}
			}
		}
		
		// Match material to mesh using material path (use cached map for faster lookup)
		if (bound_material) {
			String material_name = String(bound_material.GetPrim().GetName().GetText());
			if (material_name_to_index.has(material_name)) {
				GLTFMaterialIndex mat_i = material_name_to_index[material_name];
				Ref<Material> mat = p_state->materials[mat_i];
				if (mat.is_valid()) {
					// Assign material to mesh using instance_materials
					TypedArray<Material> instance_materials = gltf_mesh->get_instance_materials();
					if (instance_materials.is_empty()) {
						// Initialize with null materials for all surfaces
						for (int surf_i = 0; surf_i < import_mesh->get_surface_count(); surf_i++) {
							instance_materials.append(Ref<Material>());
						}
					}
					// Assign material to first surface (USD typically has one material per mesh)
					if (instance_materials.size() > 0) {
						instance_materials[0] = mat;
					}
					gltf_mesh->set_instance_materials(instance_materials);
				}
			}
		}

		gltf_mesh->set_mesh(import_mesh);
		p_state->meshes.push_back(gltf_mesh);
		path_to_mesh_index[prim.GetPath()] = mesh_index;

		// Find corresponding node and set mesh index using cached mapping
		SdfPath prim_path = prim.GetPath();
		String prim_path_str = String(prim_path.GetText());
		
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

Error USDDocument::_parse_materials(Ref<USDState> p_state) {
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		ERR_PRINT("USD: Invalid stage in _parse_materials");
		return ERR_INVALID_DATA;
	}

	HashMap<SdfPath, GLTFMaterialIndex> material_path_to_index;
	
	// First pass: collect all materials
	UsdPrimRange range = stage->Traverse();
	for (UsdPrim prim : range) {
		UsdShadeMaterial material(prim);
		if (material) {
			Ref<StandardMaterial3D> godot_material;
			godot_material.instantiate();
			godot_material->set_name(String(prim.GetName().GetText()));
			godot_material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
			
			GLTFMaterialIndex material_index = p_state->materials.size();
			material_path_to_index[prim.GetPath()] = material_index;
			
			// Get the surface output
			UsdShadeOutput surface_output = material.GetSurfaceOutput();
			if (surface_output) {
				// Get connected shader
				UsdShadeConnectableAPI source;
				TfToken source_name;
				UsdShadeAttributeType source_type;
				if (surface_output.GetConnectedSource(&source, &source_name, &source_type)) {
					UsdShadeShader surface_shader = UsdShadeShader::Get(stage, source.GetPath());
					if (surface_shader) {
						TfToken id;
						if (surface_shader.GetIdAttr().Get(&id) && id == TfToken("UsdPreviewSurface")) {
							// Parse baseColor (albedo)
							UsdShadeInput base_color_input = surface_shader.GetInput(TfToken("diffuseColor"));
							if (!base_color_input) {
								base_color_input = surface_shader.GetInput(TfToken("baseColor"));
							}
							if (base_color_input) {
								// Check if connected to texture
								UsdShadeConnectableAPI texture_source;
								TfToken texture_source_name;
								UsdShadeAttributeType texture_source_type;
								if (base_color_input.GetConnectedSource(&texture_source, &texture_source_name, &texture_source_type)) {
									UsdShadeShader texture_shader = UsdShadeShader::Get(stage, texture_source.GetPath());
									if (texture_shader) {
										TfToken texture_id;
										if (texture_shader.GetIdAttr().Get(&texture_id) && texture_id == TfToken("UsdUVTexture")) {
											UsdShadeInput file_input = texture_shader.GetInput(TfToken("file"));
											if (file_input) {
												SdfAssetPath asset_path;
												if (file_input.Get(&asset_path)) {
													String texture_path = String(asset_path.GetResolvedPath().c_str());
													if (texture_path.is_empty()) {
														texture_path = String(asset_path.GetAssetPath().c_str());
													}
													// Find texture index
													for (GLTFTextureIndex tex_i = 0; tex_i < p_state->textures.size(); tex_i++) {
														// Match by path - we'd need to store paths, for now use simple matching
														Ref<Texture2D> tex = _get_texture(p_state, tex_i, TEXTURE_TYPE_GENERIC);
														if (tex.is_valid() && tex->get_path().contains(texture_path.get_file())) {
															godot_material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, tex);
															break;
														}
													}
												}
											}
										}
									}
								} else {
									// Get color value directly
									GfVec3f color_value;
									if (base_color_input.Get(&color_value)) {
										Color albedo(color_value[0], color_value[1], color_value[2]);
										godot_material->set_albedo(albedo.linear_to_srgb());
									}
								}
							}
							
							// Parse metallic
							UsdShadeInput metallic_input = surface_shader.GetInput(TfToken("metallic"));
							if (metallic_input) {
								float metallic_value;
								if (metallic_input.Get(&metallic_value)) {
									godot_material->set_metallic(metallic_value);
								}
							}
							
							// Parse roughness
							UsdShadeInput roughness_input = surface_shader.GetInput(TfToken("roughness"));
							if (roughness_input) {
								float roughness_value;
								if (roughness_input.Get(&roughness_value)) {
									godot_material->set_roughness(roughness_value);
								}
							}
							
							// Parse normal map
							UsdShadeInput normal_input = surface_shader.GetInput(TfToken("normal"));
							if (normal_input) {
								UsdShadeConnectableAPI normal_source;
								TfToken normal_source_name;
								UsdShadeAttributeType normal_source_type;
								if (normal_input.GetConnectedSource(&normal_source, &normal_source_name, &normal_source_type)) {
									UsdShadeShader normal_shader = UsdShadeShader::Get(stage, normal_source.GetPath());
									if (normal_shader) {
										TfToken normal_id;
										if (normal_shader.GetIdAttr().Get(&normal_id) && normal_id == TfToken("UsdUVTexture")) {
											UsdShadeInput file_input = normal_shader.GetInput(TfToken("file"));
											if (file_input) {
												SdfAssetPath asset_path;
												if (file_input.Get(&asset_path)) {
													String texture_path = String(asset_path.GetResolvedPath().c_str());
													if (texture_path.is_empty()) {
														texture_path = String(asset_path.GetAssetPath().c_str());
													}
													// Find texture index
													for (GLTFTextureIndex tex_i = 0; tex_i < p_state->textures.size(); tex_i++) {
														Ref<Texture2D> tex = _get_texture(p_state, tex_i, TEXTURE_TYPE_NORMAL);
														if (tex.is_valid() && tex->get_path().contains(texture_path.get_file())) {
															godot_material->set_texture(BaseMaterial3D::TEXTURE_NORMAL, tex);
															godot_material->set_feature(BaseMaterial3D::FEATURE_NORMAL_MAPPING, true);
															break;
														}
													}
												}
											}
										}
									}
								}
							}
							
							// Parse emissive
							UsdShadeInput emissive_input = surface_shader.GetInput(TfToken("emissiveColor"));
							if (emissive_input) {
								UsdShadeConnectableAPI emissive_source;
								TfToken emissive_source_name;
								UsdShadeAttributeType emissive_source_type;
								if (emissive_input.GetConnectedSource(&emissive_source, &emissive_source_name, &emissive_source_type)) {
									UsdShadeShader emissive_shader = UsdShadeShader::Get(stage, emissive_source.GetPath());
									if (emissive_shader) {
										TfToken emissive_id;
										if (emissive_shader.GetIdAttr().Get(&emissive_id) && emissive_id == TfToken("UsdUVTexture")) {
											UsdShadeInput file_input = emissive_shader.GetInput(TfToken("file"));
											if (file_input) {
												SdfAssetPath asset_path;
												if (file_input.Get(&asset_path)) {
													String texture_path = String(asset_path.GetResolvedPath().c_str());
													if (texture_path.is_empty()) {
														texture_path = String(asset_path.GetAssetPath().c_str());
													}
													// Find texture index
													for (GLTFTextureIndex tex_i = 0; tex_i < p_state->textures.size(); tex_i++) {
														Ref<Texture2D> tex = _get_texture(p_state, tex_i, TEXTURE_TYPE_GENERIC);
														if (tex.is_valid() && tex->get_path().contains(texture_path.get_file())) {
															godot_material->set_texture(BaseMaterial3D::TEXTURE_EMISSION, tex);
															godot_material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
															break;
														}
													}
												}
											}
										}
									}
								} else {
									GfVec3f emissive_value;
									if (emissive_input.Get(&emissive_value)) {
										Color emissive(emissive_value[0], emissive_value[1], emissive_value[2]);
										godot_material->set_emission(emissive.linear_to_srgb());
										godot_material->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
									}
								}
							}
							
							// Parse opacity
							UsdShadeInput opacity_input = surface_shader.GetInput(TfToken("opacity"));
							if (opacity_input) {
								float opacity_value;
								if (opacity_input.Get(&opacity_value)) {
									if (opacity_value < 1.0f) {
										godot_material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
										Color albedo = godot_material->get_albedo();
										albedo.a = opacity_value;
										godot_material->set_albedo(albedo);
									}
								}
							}
						}
					}
				}
			}
			
			p_state->materials.push_back(godot_material);
			material_path_to_index[prim.GetPath()] = p_state->materials.size() - 1;
		}
	}

	print_verbose("USD: Total materials: " + itos(p_state->materials.size()));

	return OK;
}

Error USDDocument::_parse_skins(Ref<USDState> p_state) {
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		return ERR_INVALID_DATA;
	}

	HashMap<GLTFNodeIndex, bool> joint_mapping;
	HashMap<SdfPath, GLTFSkeletonIndex> skeleton_path_to_index;
	
	// First pass: find all skeletons and create skeleton nodes
	UsdPrimRange range = stage->Traverse();
	for (UsdPrim prim : range) {
		UsdSkelSkeleton usd_skeleton(prim);
		if (usd_skeleton) {
			GLTFSkeletonIndex skeleton_index = p_state->skeletons.size();
			skeleton_path_to_index[prim.GetPath()] = skeleton_index;
			
			Ref<GLTFSkeleton> skeleton;
			skeleton.instantiate();
			skeleton->set_name(String(prim.GetName().GetText()));
			
			// Get joints
			VtTokenArray joints;
			usd_skeleton.GetJointsAttr().Get(&joints);
			
			// Get bind transforms
			VtMatrix4dArray bind_transforms;
			usd_skeleton.GetBindTransformsAttr().Get(&bind_transforms);
			
			// Get rest transforms (if available)
			VtMatrix4dArray rest_transforms;
			usd_skeleton.GetRestTransformsAttr().Get(&rest_transforms);
			
			// Create a skin for this skeleton
			Ref<GLTFSkin> skin;
			skin.instantiate();
			skin->set_name(String(prim.GetName().GetText()) + "_skin");
			
			// Map joint paths to node indices
			HashMap<String, GLTFNodeIndex> joint_path_to_node;
			for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); node_i++) {
				Ref<GLTFNode> node = p_state->nodes[node_i];
				if (node->has_additional_data("USD_prim_path")) {
					String stored_path = node->get_additional_data("USD_prim_path");
					joint_path_to_node[stored_path] = node_i;
				}
			}
			
			// Process joints
			for (size_t joint_i = 0; joint_i < joints.size(); joint_i++) {
				String joint_path_str = String(joints[joint_i].GetText());
				// Build full path relative to skeleton
				SdfPath joint_path = prim.GetPath().AppendChild(TfToken(joint_path_str.get_file().get_basename()));
				// Try to find the joint node
				GLTFNodeIndex joint_node_index = -1;
				
				// Search for node matching this joint path
				for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); node_i++) {
					Ref<GLTFNode> node = p_state->nodes[node_i];
					if (node->has_additional_data("USD_prim_path")) {
						String stored_path = node->get_additional_data("USD_prim_path");
						// Check if the stored path ends with the joint name
						if (stored_path.ends_with(joint_path_str) || stored_path.ends_with("/" + joint_path_str)) {
							joint_node_index = node_i;
							break;
						}
					}
				}
				
				if (joint_node_index >= 0) {
					skin->joints.push_back(joint_node_index);
					skin->joints_original.push_back(joint_node_index);
					p_state->nodes.write[joint_node_index]->joint = true;
					joint_mapping[joint_node_index] = true;
					
					// Set inverse bind transform
					if (joint_i < bind_transforms.size()) {
						Transform3D bind_xform = _as_xform(bind_transforms[joint_i]);
						skin->inverse_binds.push_back(bind_xform.affine_inverse());
					} else {
						// Default identity
						skin->inverse_binds.push_back(Transform3D());
					}
					
					// Set rest transform if available
					if (joint_i < rest_transforms.size()) {
						Transform3D rest_xform = _as_xform(rest_transforms[joint_i]);
						p_state->nodes.write[joint_node_index]->set_additional_data("GODOT_rest_transform", rest_xform);
					}
				}
			}
			
			if (skin->joints.size() > 0) {
				p_state->skins.push_back(skin);
				skeleton->set_skin(p_state->skins.size() - 1);
			}
			
			p_state->skeletons.push_back(skeleton);
		}
	}
	
	// Second pass: find skinned meshes and bind them to skeletons
	for (UsdPrim prim : range) {
		UsdGeomMesh usd_mesh(prim);
		if (!usd_mesh) {
			continue;
		}
		
		// Check for SkelBindingAPI
		UsdSkelBindingAPI binding_api(prim);
		if (binding_api) {
			// Get skeleton binding
			UsdRelationship skeleton_rel = binding_api.GetSkeletonRel();
			SdfPathVector targets;
			if (skeleton_rel.GetTargets(&targets) && targets.size() > 0) {
				SdfPath skeleton_path = targets[0];
				if (skeleton_path_to_index.has(skeleton_path)) {
					GLTFSkeletonIndex skeleton_index = skeleton_path_to_index[skeleton_path];
					
					// Get joint influences from primvars
					UsdGeomPrimvar joint_indices_primvar = binding_api.GetJointIndicesPrimvar();
					UsdGeomPrimvar joint_weights_primvar = binding_api.GetJointWeightsPrimvar();
					
					if (joint_indices_primvar && joint_weights_primvar) {
						VtIntArray joint_indices;
						VtFloatArray joint_weights;
						
						if (joint_indices_primvar.Get(&joint_indices) && joint_weights_primvar.Get(&joint_weights)) {
							// Find the mesh node and assign skin using cached mapping
							SdfPath prim_path = prim.GetPath();
							String prim_path_str = String(prim_path.GetText());
							if (p_state->prim_path_to_node_index.has(prim_path_str)) {
								GLTFNodeIndex node_index = p_state->prim_path_to_node_index[prim_path_str];
								if (node_index >= 0 && node_index < p_state->nodes.size()) {
									// Assign skin to this node
									if (skeleton_index < p_state->skeletons.size()) {
										Ref<GLTFSkeleton> skeleton = p_state->skeletons[skeleton_index];
										if (skeleton.is_valid() && skeleton->get_skin() >= 0) {
											p_state->nodes[node_index]->skin = skeleton->get_skin();
											p_state->nodes[node_index]->skeleton = skeleton_index;
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
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		return ERR_INVALID_DATA;
	}

	// Get stage time range
	double start_time = stage->GetStartTimeCode();
	double end_time = stage->GetEndTimeCode();
	if (start_time == end_time) {
		// No time range set, check for time samples
		start_time = 0.0;
		end_time = 1.0;
	}

	// First pass: find UsdSkelAnimation prims for skeletal animations
	UsdPrimRange range = stage->Traverse();
	HashMap<SdfPath, GLTFAnimationIndex> animation_path_to_index;
	
	for (UsdPrim prim : range) {
		UsdSkelAnimation usd_animation(prim);
		if (usd_animation) {
			Ref<GLTFAnimation> animation;
			animation.instantiate();
			animation->set_name(String(prim.GetName().GetText()));
			animation->set_original_name(String(prim.GetName().GetText()));
			
			// Get time samples from individual attributes
			std::vector<double> time_samples;
			UsdAttribute translations_attr = usd_animation.GetTranslationsAttr();
			UsdAttribute rotations_attr = usd_animation.GetRotationsAttr();
			UsdAttribute scales_attr = usd_animation.GetScalesAttr();
			
			// Collect time samples from all transform attributes
			std::vector<double> trans_times, rot_times, scale_times;
			if (translations_attr) {
				translations_attr.GetTimeSamples(&trans_times);
			}
			if (rotations_attr) {
				rotations_attr.GetTimeSamples(&rot_times);
			}
			if (scales_attr) {
				scales_attr.GetTimeSamples(&scale_times);
			}
			
			// Union all time samples
			time_samples.insert(time_samples.end(), trans_times.begin(), trans_times.end());
			time_samples.insert(time_samples.end(), rot_times.begin(), rot_times.end());
			time_samples.insert(time_samples.end(), scale_times.begin(), scale_times.end());
			
			// Remove duplicates and sort
			std::sort(time_samples.begin(), time_samples.end());
			time_samples.erase(std::unique(time_samples.begin(), time_samples.end()), time_samples.end());
			
			// If no time samples, create a single frame animation
			if (time_samples.empty()) {
				time_samples.push_back(start_time);
			}
			
			// Get joints that this animation affects
			VtTokenArray joints;
			usd_animation.GetJointsAttr().Get(&joints);
			
			// Process each time sample
			for (double time : time_samples) {
				UsdTimeCode time_code(time);
				
				// Get translations, rotations, scales
				VtVec3fArray translations;
				VtQuatfArray rotations;
				VtVec3hArray scales;
				
				if (usd_animation.GetTranslationsAttr().Get(&translations, time_code) &&
					usd_animation.GetRotationsAttr().Get(&rotations, time_code) &&
					usd_animation.GetScalesAttr().Get(&scales, time_code)) {
					
					// Match joints to nodes
					for (size_t joint_i = 0; joint_i < joints.size() && joint_i < translations.size(); joint_i++) {
						String joint_name = String(joints[joint_i].GetText());
						
						// Find node matching this joint
						GLTFNodeIndex node_index = -1;
						for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); node_i++) {
							Ref<GLTFNode> node = p_state->nodes[node_i];
							if (node->has_additional_data("USD_prim_path")) {
								String stored_path = node->get_additional_data("USD_prim_path");
								if (stored_path.ends_with(joint_name) || stored_path.ends_with("/" + joint_name)) {
									node_index = node_i;
									break;
								}
							}
						}
						
						if (node_index >= 0) {
							GLTFAnimation::NodeTrack &track = animation->get_node_tracks()[node_index];
							
							// Add translation key
							if (joint_i < translations.size()) {
								track.position_track.times.push_back(time);
								track.position_track.values.push_back(_as_vec3(translations[joint_i]));
							}
							
							// Add rotation key
							if (joint_i < rotations.size()) {
								Quaternion rot(rotations[joint_i].GetReal(), rotations[joint_i].GetImaginary()[0],
											   rotations[joint_i].GetImaginary()[1], rotations[joint_i].GetImaginary()[2]);
								track.rotation_track.times.push_back(time);
								track.rotation_track.values.push_back(rot);
							}
							
							// Add scale key
							if (joint_i < scales.size()) {
								track.scale_track.times.push_back(time);
								track.scale_track.values.push_back(Vector3(scales[joint_i][0], scales[joint_i][1], scales[joint_i][2]));
							}
						}
					}
				}
			}
			
			if (animation->get_node_tracks().size() > 0) {
				GLTFAnimationIndex anim_index = p_state->animations.size();
				p_state->animations.push_back(animation);
				animation_path_to_index[prim.GetPath()] = anim_index;
			}
		}
	}
	
	// Second pass: find transform animations on regular prims
	for (UsdPrim prim : range) {
		UsdGeomXformable xformable(prim);
		if (!xformable) {
			continue;
		}
		
		// Check if transform attributes have time samples
		std::vector<double> time_samples;
		UsdAttribute xform_attr = xformable.GetXformOpOrderAttr();
		if (xform_attr) {
			xform_attr.GetTimeSamples(&time_samples);
		}
		
		// Also check individual xform ops
		if (time_samples.empty()) {
			UsdGeomXformOp::Type op_types[] = {
				UsdGeomXformOp::TypeTranslate,
				UsdGeomXformOp::TypeRotateXYZ,
				UsdGeomXformOp::TypeScale
			};
			
			for (UsdGeomXformOp::Type op_type : op_types) {
				UsdGeomXformOp op = xformable.GetXformOp(op_type);
				if (op) {
					UsdAttribute attr = op.GetAttr();
					if (attr) {
						std::vector<double> op_times;
						attr.GetTimeSamples(&op_times);
						time_samples.insert(time_samples.end(), op_times.begin(), op_times.end());
					}
				}
			}
		}
		
		if (!time_samples.empty()) {
			// Find the node for this prim using cached mapping
			SdfPath prim_path = prim.GetPath();
			String prim_path_str = String(prim_path.GetText());
			GLTFNodeIndex node_index = -1;
			if (p_state->prim_path_to_node_index.has(prim_path_str)) {
				node_index = p_state->prim_path_to_node_index[prim_path_str];
			}
			
			if (node_index >= 0) {
				// Create or get animation for this node
				Ref<GLTFAnimation> animation;
				String anim_name = String(prim.GetName().GetText()) + "_transform";
				
				// Check if we already have an animation for this node
				GLTFAnimationIndex existing_anim_index = -1;
				for (GLTFAnimationIndex anim_i = 0; anim_i < p_state->animations.size(); anim_i++) {
					Ref<GLTFAnimation> existing_anim = p_state->animations[anim_i];
					if (existing_anim->get_name() == anim_name) {
						animation = existing_anim;
						existing_anim_index = anim_i;
						break;
					}
				}
				
				if (animation.is_null()) {
					animation.instantiate();
					animation->set_name(anim_name);
					animation->set_original_name(anim_name);
				}
				
				GLTFAnimation::NodeTrack &track = animation->get_node_tracks()[node_index];
				
				// Sample transforms at each time
				for (double time : time_samples) {
					UsdTimeCode time_code(time);
					
					GfMatrix4d local_transform;
					bool reset_xform_stack = false;
					xformable.GetLocalTransformation(&local_transform, &reset_xform_stack, time_code);
					
					Transform3D xform = _as_xform(local_transform);
					
					track.position_track.times.push_back(time);
					track.position_track.values.push_back(xform.origin);
					
					track.rotation_track.times.push_back(time);
					track.rotation_track.values.push_back(xform.basis.get_rotation_quaternion());
					
					Vector3 scale = xform.basis.get_scale();
					track.scale_track.times.push_back(time);
					track.scale_track.values.push_back(scale);
				}
				
				if (existing_anim_index < 0 && track.position_track.times.size() > 0) {
					p_state->animations.push_back(animation);
				}
			}
		}
	}

	print_verbose("USD: Total animations: " + itos(p_state->animations.size()));

	return OK;
}

Error USDDocument::_parse_cameras(Ref<USDState> p_state) {
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		return ERR_INVALID_DATA;
	}

	HashMap<SdfPath, GLTFCameraIndex> camera_path_to_index;
	
	UsdPrimRange range = stage->Traverse();
	for (UsdPrim prim : range) {
		UsdGeomCamera usd_camera(prim);
		if (usd_camera) {
			GLTFCameraIndex camera_index = p_state->cameras.size();
			camera_path_to_index[prim.GetPath()] = camera_index;
			
			Ref<GLTFCamera> camera;
			camera.instantiate();
			camera->set_name(String(prim.GetName().GetText()));
			
			// Get projection type
			TfToken projection;
			usd_camera.GetProjectionAttr().Get(&projection);
			bool is_perspective = (projection == TfToken("perspective"));
			camera->set_perspective(is_perspective);
			
			if (is_perspective) {
				// Calculate FOV from focal length and horizontal aperture
				double focal_length = 50.0; // Default
				double horizontal_aperture = 20.955; // Default (inches)
				usd_camera.GetFocalLengthAttr().Get(&focal_length);
				usd_camera.GetHorizontalApertureAttr().Get(&horizontal_aperture);
				
				// Convert to FOV in radians
				// FOV = 2 * atan(horizontal_aperture / (2 * focal_length))
				// USD uses mm for focal length and inches for aperture, convert to same units
				double horizontal_aperture_mm = horizontal_aperture * 25.4; // inches to mm
				double fov_rad = 2.0 * atan(horizontal_aperture_mm / (2.0 * focal_length));
				camera->set_fov(fov_rad);
			} else {
				// Orthographic camera
				double horizontal_aperture = 20.955;
				usd_camera.GetHorizontalApertureAttr().Get(&horizontal_aperture);
				camera->set_size_mag(horizontal_aperture * 0.5 * 25.4); // Convert inches to mm, then half for size
			}
			
			// Get near and far planes
			double near_plane = 0.1;
			double far_plane = 1000.0;
			usd_camera.GetClippingRangeAttr().Get(&near_plane, &far_plane);
			camera->set_depth_near(near_plane);
			camera->set_depth_far(far_plane);
			
			p_state->cameras.push_back(camera);
			
			// Find corresponding node and set camera index using cached mapping
			SdfPath prim_path = prim.GetPath();
			String prim_path_str = String(prim_path.GetText());
			if (p_state->prim_path_to_node_index.has(prim_path_str)) {
				GLTFNodeIndex node_index = p_state->prim_path_to_node_index[prim_path_str];
				if (node_index >= 0 && node_index < p_state->nodes.size()) {
					p_state->nodes[node_index]->camera = camera_index;
				}
			}
		}
	}

	print_verbose("USD: Total cameras: " + itos(p_state->cameras.size()));

	return OK;
}

Error USDDocument::_parse_lights(Ref<USDState> p_state) {
	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		return ERR_INVALID_DATA;
	}

	// USD uses UsdLux for lights
	UsdPrimRange range = stage->Traverse();
	for (UsdPrim prim : range) {
		// Check for various light types
		if (prim.IsA<pxr::UsdLuxLight>()) {
			pxr::UsdLuxLight usd_light(prim);
			
			Ref<GLTFLight> light;
			light.instantiate();
			light->set_name(String(prim.GetName().GetText()));
			
			// Get color
			GfVec3f color_value(1.0f, 1.0f, 1.0f);
			usd_light.GetColorAttr().Get(&color_value);
			light->set_color(Color(color_value[0], color_value[1], color_value[2]));
			
			// Get intensity
			float intensity = 1.0f;
			usd_light.GetIntensityAttr().Get(&intensity);
			light->set_intensity(intensity);
			
			// Determine light type based on prim type
			String prim_type = String(prim.GetTypeName().GetText());
			if (prim_type.contains("DistantLight") || prim_type == "DistantLight") {
				light->set_light_type("directional");
			} else if (prim_type.contains("RectLight") || prim_type == "RectLight") {
				light->set_light_type("area");
			} else if (prim_type.contains("DiskLight") || prim_type == "DiskLight") {
				light->set_light_type("area");
			} else if (prim_type.contains("SphereLight") || prim_type == "SphereLight") {
				light->set_light_type("point");
			} else if (prim_type.contains("CylinderLight") || prim_type == "CylinderLight") {
				light->set_light_type("point");
			} else {
				// Default to point light
				light->set_light_type("point");
			}
			
			// For spot lights, check if it's a UsdLuxShapingAPI
			pxr::UsdLuxShapingAPI shaping_api(prim);
			if (shaping_api) {
				double cone_angle = 0.0;
				if (shaping_api.GetShapingConeAngleAttr().Get(&cone_angle) && cone_angle > 0.0) {
					light->set_light_type("spot");
					light->set_outer_cone_angle(Math::deg_to_rad(cone_angle));
					
					double cone_softness = 0.0;
					if (shaping_api.GetShapingConeSoftnessAttr().Get(&cone_softness)) {
						double inner_angle = cone_angle * (1.0 - cone_softness);
						light->set_inner_cone_angle(Math::deg_to_rad(inner_angle));
					}
				}
			}
			
			p_state->lights.push_back(light);
			
			// Find corresponding node and set light index using cached mapping
			SdfPath prim_path = prim.GetPath();
			String prim_path_str = String(prim_path.GetText());
			if (p_state->prim_path_to_node_index.has(prim_path_str)) {
				GLTFNodeIndex node_index = p_state->prim_path_to_node_index[prim_path_str];
				if (node_index >= 0 && node_index < p_state->nodes.size()) {
					p_state->nodes[node_index]->light = p_state->lights.size() - 1;
				}
			}
		}
	}

	print_verbose("USD: Total lights: " + itos(p_state->lights.size()));

	return OK;
}

Error USDDocument::_parse_images(Ref<USDState> p_state, const String &p_base_path) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		return ERR_INVALID_DATA;
	}

	HashSet<String> texture_paths;
	
	// Collect texture paths from materials
	UsdPrimRange range = stage->Traverse();
	for (UsdPrim prim : range) {
		UsdShadeMaterial material(prim);
		if (material) {
			// Get the surface output
			UsdShadeOutput surface_output = material.GetSurfaceOutput();
			if (surface_output) {
				// Get connected shader
				UsdShadeShader surface_shader = UsdShadeShader::Get(stage, surface_output.GetConnectedSource().source.GetPath());
				if (surface_shader) {
					// Check for texture inputs in UsdPreviewSurface
					// Common texture inputs: diffuseColor, emissiveColor, normal, metallicRoughness, occlusion
					Vector<TfToken> texture_inputs = {
						TfToken("diffuseColor"),
						TfToken("emissiveColor"),
						TfToken("normal"),
						TfToken("metallicRoughness"),
						TfToken("occlusion"),
						TfToken("baseColor"), // Alternative name
					};

					for (const TfToken &input_name : texture_inputs) {
						UsdShadeInput input = surface_shader.GetInput(input_name);
						if (input) {
							// Check if connected to a texture shader
							UsdShadeConnectableAPI source;
							TfToken source_name;
							UsdShadeAttributeType source_type;
							if (input.GetConnectedSource(&source, &source_name, &source_type)) {
								UsdShadeShader texture_shader = UsdShadeShader::Get(stage, source.GetPath());
								if (texture_shader) {
									// Check if it's a UsdUVTexture
									TfToken id;
									if (texture_shader.GetIdAttr().Get(&id) && id == TfToken("UsdUVTexture")) {
										UsdShadeInput file_input = texture_shader.GetInput(TfToken("file"));
										if (file_input) {
											SdfAssetPath asset_path;
											if (file_input.Get(&asset_path)) {
												String resolved_path = String(asset_path.GetResolvedPath().c_str());
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
	// TODO: Implement
	return nullptr;
}

ImporterMeshInstance3D *USDDocument::_generate_mesh_instance(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement
	return nullptr;
}

Camera3D *USDDocument::_generate_camera(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement
	return nullptr;
}

Light3D *USDDocument::_generate_light(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement
	return nullptr;
}

Node3D *USDDocument::_generate_spatial(Ref<USDState> p_state, const GLTFNodeIndex p_node_index) {
	// TODO: Implement
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

