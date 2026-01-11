/**************************************************************************/
/*  merge.cpp                                                             */
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

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/image.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "editor/editor_node.h"
#include "modules/scene_merge/mesh_merge_triangle.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/main/node.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"

#include "thirdparty/misc/rjm_texbleed.h"
#include "thirdparty/xatlas/xatlas.h"
#include <cmath>
#include <cstdint>

#include "merge.h"

bool MeshTextureAtlas::set_atlas_texel(void *param, int x, int y, const Vector3 &bar, const Vector3 &, const Vector3 &, float) {
	ERR_FAIL_NULL_V(param, false);
	AtlasTextureArguments *args = static_cast<AtlasTextureArguments *>(param);
	ERR_FAIL_NULL_V(args, false);
	if (args->source_texture.is_valid()) {
		const Vector2 source_uv = interpolate_source_uvs(bar, args);
		Pair<int, int> coordinates = calculate_coordinates(source_uv, args->source_texture->get_width(), args->source_texture->get_height());
		const Color color = args->source_texture->get_pixel(coordinates.first, coordinates.second);
		args->atlas_data->set_pixel(x, y, color);
		int32_t index = y * args->atlas_width + x;
		AtlasLookupTexel &lookup = args->atlas_lookup[index];
		lookup.material_index = args->material_index;
		lookup.x = static_cast<uint16_t>(coordinates.first);
		lookup.y = static_cast<uint16_t>(coordinates.second);
		return true;
	}
	return false;
}

void MeshTextureAtlas::_find_all_mesh_instances(Vector<MeshMerge> &r_items, Node *p_current_node, const Node *p_owner) {
	if (!p_current_node) {
		return;
	}

	ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_current_node);
	if (mi && mi->get_mesh().is_valid()) {
		Ref<ImporterMesh> importer_mesh = mi->get_mesh();
		for (int32_t surface_i = 0; surface_i < importer_mesh->get_surface_count(); surface_i++) {
			Ref<Material> surface_material = mi->get_surface_material(surface_i);
			if (!surface_material.is_valid()) {
				surface_material = importer_mesh->get_surface_material(surface_i);
			}

			Array array = importer_mesh->get_surface_arrays(surface_i).duplicate(true);
			MeshState mesh_state;
			mesh_state.importer_mesh = importer_mesh;
			if (mi->is_inside_tree()) {
				mesh_state.path = mi->get_path();
			}
			mesh_state.importer_mesh_instance = mi;

			if (r_items.is_empty()) {
				return;
			}
			MeshMerge &mesh = r_items.write[r_items.size() - 1];

			mesh_state.index_offset = mesh.vertex_count;
			mesh.vertex_count += PackedVector3Array(array[Mesh::ARRAY_VERTEX]).size();

			if (mesh_state.is_valid()) {
				mesh.meshes.push_back(mesh_state);
			}
		}
	}

	for (int32_t child_i = 0; child_i < p_current_node->get_child_count(); child_i++) {
		Node *child = p_current_node->get_child(child_i);
		if (child != p_owner) { // Add base case to stop recursion
			_find_all_mesh_instances(r_items, child, p_owner);
		}
	}
}

void MeshTextureAtlas::_bind_methods() {
	ClassDB::bind_static_method("MeshTextureAtlas", D_METHOD("merge", "root"), &MeshTextureAtlas::merge_meshes);
}

/**
 * Validates input parameters and returns early if invalid.
 * Following Elixir Credo: keep validation logic separate and explicit.
 */
static bool validate_merge_input(Node *root_node) {
	if (root_node == nullptr) {
		ERR_PRINT("SceneMerge: Cannot merge meshes with null root node");
		return false;
	}
	return true;
}

/**
 * Collects all valid mesh instances from the scene hierarchy.
 * Returns the count of meshes found.
 */
static size_t collect_mesh_instances(Node *root_node, Vector<MeshTextureAtlas::MeshMerge> &mesh_merges) {
	MeshTextureAtlas::find_all_mesh_instances(mesh_merges, root_node, nullptr);
	return mesh_merges[0].meshes.size();
}

/**
 * Processes and transforms the geometry from a single mesh instance into the surface tool.
 * This function handles the core mesh processing logic during merging.
 *
 * Processing steps:
 * 1. Extract vertex, normal, and index data from each mesh surface
 * 2. Apply the mesh instance's global transform to all vertices and normals
 * 3. Set up surface materials in the surface tool
 * 4. Add transformed vertices and normals to the surface tool
 * 5. Handle indexed geometry by adding triangle indices in groups of 3
 */
static void process_mesh_geometry(const MeshTextureAtlas::MeshState &p_mesh_state, const Ref<SurfaceTool> &p_surface_tool) {
	const Ref<ImporterMesh> mesh = p_mesh_state.importer_mesh;
	const ImporterMeshInstance3D *mesh_instance = p_mesh_state.importer_mesh_instance;

	// Early return for invalid inputs - tail call optimization
	if (mesh.is_null() || mesh_instance == nullptr) {
		return; // Tail call
	}

	// Get the complete transformation from this mesh instance's position in the scene
	// In headless/testing environments, the node may not be marked as inside_tree yet
	Transform3D global_transform;
	if (mesh_instance->is_inside_tree()) {
		global_transform = mesh_instance->get_global_transform();
	} else {
		// Calculate global transform manually by walking up the tree
		Node3D *node_3d = Object::cast_to<Node3D>((Node3D *)mesh_instance);
		global_transform = node_3d ? node_3d->get_transform() : Transform3D();
		Node *current = mesh_instance->get_parent();
		while (current) {
			Node3D *parent_3d = Object::cast_to<Node3D>(current);
			if (parent_3d) {
				global_transform = parent_3d->get_transform() * global_transform;
			}
			current = current->get_parent();
		}
	}

	// Process each material surface in the mesh (meshes can have multiple materials)
	for (int surface_index = 0; surface_index < mesh->get_surface_count(); surface_index++) {
		// Get the raw geometry data for this surface
		const Array surface_arrays = mesh->get_surface_arrays(surface_index);
		const Ref<Material> surface_material = mesh->get_surface_material(surface_index);

		// Extract vertex and normal arrays from the surface data
		Vector<Vector3> vertices = surface_arrays[Mesh::ARRAY_VERTEX];
		Vector<Vector3> normals = surface_arrays[Mesh::ARRAY_NORMAL];
		const Vector<int> indices = surface_arrays[Mesh::ARRAY_INDEX];

		// Apply the global transformation to place this mesh correctly in world space
		for (int vertex_index = 0; vertex_index < vertices.size(); vertex_index++) {
			// Transform vertex position (translation + rotation + scale)
			vertices.write[vertex_index] = global_transform.xform(vertices[vertex_index]);

			// Transform normal vectors (rotation only - normals don't translate)
			if (vertex_index < normals.size()) {
				normals.write[vertex_index] = global_transform.basis.xform(normals[vertex_index]);
			}
		}

		// Set up the surface tool for this material group
		p_surface_tool->set_material(surface_material);

		// Add all vertices and their corresponding normals to the surface tool
		for (int vertex_index = 0; vertex_index < vertices.size(); vertex_index++) {
			// Set normal first, then vertex (SurfaceTool expects this order)
			if (vertex_index < normals.size()) {
				p_surface_tool->set_normal(normals[vertex_index]);
			}
			p_surface_tool->add_vertex(vertices[vertex_index]);
		}

		// Handle triangle indices if this is an indexed mesh
		if (!indices.is_empty()) {
			// Indices come in groups of 3 (one triangle each)
			// Each triangle is defined by 3 vertex indices
			// CRITICAL FIX: Add vertex offset for multi-mesh merging (indices from second mesh need to account for vertices from first mesh)
			int32_t vertex_offset = p_mesh_state.index_offset;
			for (size_t triangle_start = 0; triangle_start < indices.size(); triangle_start += 3) {
				p_surface_tool->add_index(indices[triangle_start] + vertex_offset); // First vertex of triangle
				p_surface_tool->add_index(indices[triangle_start + 1] + vertex_offset); // Second vertex of triangle
				p_surface_tool->add_index(indices[triangle_start + 2] + vertex_offset); // Third vertex of triangle
			}
		}
	}
}

/**
 * Removes original mesh instances from the scene hierarchy.
 * Tail call optimized with direct returns for invalid cases.
 */
static void remove_original_instances(const Vector<MeshTextureAtlas::MeshState> &p_mesh_states) {
	for (const MeshTextureAtlas::MeshState &mesh_state : p_mesh_states) {
		ImporterMeshInstance3D *mesh_instance = mesh_state.importer_mesh_instance;
		if (mesh_instance == nullptr) {
			continue; // Tail call - continue loop
		}

		Node *parent = mesh_instance->get_parent();
		if (parent == nullptr) {
			continue; // Tail call - continue loop
		}

		parent->remove_child(mesh_instance);
		mesh_instance->queue_free();
	}
}

// Magic numbers refactored to named constants for better readability
static constexpr int64_t MINIMUM_MESH_COUNT = 2;

/**
 * Builds a mesh with proper blend shape support by directly using ImporterMesh
 * This ensures blend shape vertex data is properly preserved and transformed
 */
static Ref<ImporterMesh> build_mesh_with_blend_shapes(const Vector<MeshTextureAtlas::MeshState> &p_mesh_states, const Vector<String> &p_blend_shape_names) {
	Ref<ImporterMesh> result_mesh;
	result_mesh.instantiate();
	result_mesh->set_name("MergedMeshWithBlendShapes");

	const int blend_shape_count = p_blend_shape_names.size();

	// Add blend shapes first (required before surfaces)
	for (int bs_i = 0; bs_i < blend_shape_count; bs_i++) {
		result_mesh->add_blend_shape(p_blend_shape_names[bs_i]);
	}

	int global_vertex_offset = 0;

	for (const MeshTextureAtlas::MeshState &mesh_state : p_mesh_states) {
		Ref<ImporterMesh> input_mesh = mesh_state.importer_mesh;

		// Get global transform
		Transform3D global_transform;
		if (mesh_state.importer_mesh_instance->is_inside_tree()) {
			global_transform = mesh_state.importer_mesh_instance->get_global_transform();
		} else {
			Node3D *node_3d = Object::cast_to<Node3D>((Node3D *)mesh_state.importer_mesh_instance);
			global_transform = node_3d ? node_3d->get_transform() : Transform3D();
			Node *current = mesh_state.importer_mesh_instance->get_parent();
			while (current) {
				Node3D *parent_3d = Object::cast_to<Node3D>(current);
				if (parent_3d) {
					global_transform = parent_3d->get_transform() * global_transform;
				}
				current = current->get_parent();
			}
		}

		for (int surface_idx = 0; surface_idx < input_mesh->get_surface_count(); surface_idx++) {
			Array surface_arrays = input_mesh->get_surface_arrays(surface_idx);
			Ref<Material> surface_material = input_mesh->get_surface_material(surface_idx);

			// Transform base geometry
			Vector<Vector3> vertices = surface_arrays[Mesh::ARRAY_VERTEX];
			Vector<Vector3> normals = surface_arrays[Mesh::ARRAY_NORMAL];

			for (int vertex_idx = 0; vertex_idx < vertices.size(); vertex_idx++) {
				vertices.write[vertex_idx] = global_transform.xform(vertices[vertex_idx]);
				if (vertex_idx < normals.size()) {
					normals.write[vertex_idx] = global_transform.basis.xform(normals[vertex_idx]);
				}
			}

			surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
			surface_arrays[Mesh::ARRAY_NORMAL] = normals;

			// Handle index offset
			Vector<int> indices = surface_arrays[Mesh::ARRAY_INDEX];
			for (int i = 0; i < indices.size(); i++) {
				indices.write[i] += global_vertex_offset;
			}
			surface_arrays[Mesh::ARRAY_INDEX] = indices;

			// Collect blend shape data
			TypedArray<Array> bs_arrays_for_surface;
			for (int bs_idx = 0; bs_idx < blend_shape_count; bs_idx++) {
				Array surface_bs_arrays;

				if (bs_idx < input_mesh->get_blend_shape_count()) {
					// Get blend shape data from input mesh
					surface_bs_arrays = input_mesh->get_surface_blend_shape_arrays(surface_idx, bs_idx).duplicate(true);

					// Transform blend shape vertex positions
					if (!surface_bs_arrays.is_empty() && surface_bs_arrays.size() > Mesh::ARRAY_VERTEX) {
						Vector<Vector3> bs_vertices = surface_bs_arrays[Mesh::ARRAY_VERTEX];
						for (int bs_vertex_idx = 0; bs_vertex_idx < bs_vertices.size(); bs_vertex_idx++) {
							bs_vertices.write[bs_vertex_idx] = global_transform.xform(bs_vertices[bs_vertex_idx]);
						}
						surface_bs_arrays[Mesh::ARRAY_VERTEX] = bs_vertices;
					}
				} else {
					// Create empty blend shape arrays if this mesh doesn't have this shape
					surface_bs_arrays = Array();
					surface_bs_arrays.resize(RS::ARRAY_MAX); // Use RenderingServer ARRAY_MAX
					for (int arr_idx = 0; arr_idx < RS::ARRAY_MAX; arr_idx++) {
						surface_bs_arrays[arr_idx] = Variant();
					}
				}

				bs_arrays_for_surface.append(surface_bs_arrays);
			}

			// Add surface with blend shapes
			Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
			result_mesh->add_surface(primitive, surface_arrays, bs_arrays_for_surface);
			int surface_index = result_mesh->get_surface_count() - 1;
			result_mesh->set_surface_material(surface_index, surface_material);

			global_vertex_offset += vertices.size();
		}
	}

	return result_mesh;
}

Node *MeshTextureAtlas::merge_meshes(Node *p_root) {
	// Validate input parameters first
	if (!validate_merge_input(p_root)) {
		return p_root; // Early return for invalid input
	}

	// Collect all valid mesh instances from the scene
	Vector<MeshMerge> mesh_merges;
	MeshMerge mesh_merge;
	mesh_merges.push_back(mesh_merge);

	size_t mesh_count = collect_mesh_instances(p_root, mesh_merges);

	// Handle edge cases gracefully
	if (mesh_count == 0) {
		print_line("SceneMerge: No mesh instances found in scene, nothing to merge");
		return p_root;
	}

	if (mesh_count < MINIMUM_MESH_COUNT) {
		print_line("SceneMerge: Only one mesh instance found, nothing to merge");
		return p_root;
	}

	print_line(vformat("SceneMerge: Found %d mesh instances, proceeding with merge", (int64_t)mesh_count));

	// Initialize surface tool for mesh construction
	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);

	// Process all mesh geometry
	const Vector<MeshTextureAtlas::MeshState> &mesh_states = mesh_merges[0].meshes;
	for (const MeshState &mesh_state : mesh_states) {
		process_mesh_geometry(mesh_state, surface_tool);
	}

	// Clean up original scene hierarchy
	remove_original_instances(mesh_states);

	// Handle blend shape compatibility and data merging
	bool has_blend_shapes = false;
	int blend_shape_count = 0;
	Vector<String> blend_shape_names;

	// First pass: check compatibility and collect names
	for (const MeshState &mesh_state : mesh_states) {
		Ref<ImporterMesh> input_mesh = mesh_state.importer_mesh;
		int mesh_bs_count = input_mesh->get_blend_shape_count();
		if (mesh_bs_count > 0) {
			if (!has_blend_shapes) {
				blend_shape_count = mesh_bs_count;
				has_blend_shapes = true;
				// Collect blend shape names from first mesh
				for (int bs_i = 0; bs_i < blend_shape_count; bs_i++) {
					blend_shape_names.push_back(input_mesh->get_blend_shape_name(bs_i));
				}
			} else if (blend_shape_count != mesh_bs_count) {
				ERR_PRINT("SceneMerge: Cannot merge meshes with different blend shape counts - skipping blend shape preservation");
				has_blend_shapes = false;
				blend_shape_names.clear();
				break;
			}
		}
	}

	// Create ImporterMesh directly for blend shape support
	Ref<ImporterMesh> merged_importer_mesh;
	merged_importer_mesh.instantiate();
	merged_importer_mesh->set_name("MergedMesh");

	if (has_blend_shapes && mesh_states.size() >= 1) {
		print_line(vformat("SceneMerge: Detected blend shapes in %d mesh(es) - ensuring vertex order preservation", mesh_states.size()));

		// For full blend shape preservation, construct mesh directly via ImporterMesh
		// This approach properly handles blend shape vertex data

		Ref<ImporterMesh> blend_shape_merged_mesh = build_mesh_with_blend_shapes(mesh_states, blend_shape_names);
		if (blend_shape_merged_mesh.is_valid()) {
			merged_importer_mesh = blend_shape_merged_mesh;

			print_line(vformat("SceneMerge: Successfully merged mesh with %d blend shapes preserved", merged_importer_mesh->get_blend_shape_count()));

			// Skip the regular merging, go directly to final material and cleanup
			goto material_cleanup;
		} else {
			// Fallback to regular SurfaceTool approach
			print_line("SceneMerge: Blend shape construction failed, falling back to regular merge");
			has_blend_shapes = false;
		}
	}

	print_line(vformat("SceneMerge: Final merged mesh has %d blend shapes", merged_importer_mesh->get_blend_shape_count()));

material_cleanup:
	// Simple base color merge: average all BaseMaterial3D albedo colors
	Ref<StandardMaterial3D> merged_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	merged_material->set_name("MergedMaterial");

	// Collect and average base colors from all materials
	Vector<Color> base_colors;
	for (const MeshState &mesh_state : mesh_states) {
		Ref<ImporterMesh> importer_mesh = mesh_state.importer_mesh;
		for (int surface_i = 0; surface_i < importer_mesh->get_surface_count(); surface_i++) {
			Ref<Material> surface_material = importer_mesh->get_surface_material(surface_i);
			if (surface_material.is_null()) {
				continue;
			}

			Ref<BaseMaterial3D> base_material = surface_material;
			if (base_material.is_valid()) {
				Color albedo = base_material->get_albedo();
				base_colors.push_back(albedo);
			}
		}
	}

	// Average the colors (simple merge)
	if (!base_colors.is_empty()) {
		Color averaged_color = Color(0, 0, 0, 1);
		for (const Color &color : base_colors) {
			averaged_color += color;
		}
		averaged_color /= (float)base_colors.size();
		merged_material->set_albedo(averaged_color);

		print_line(vformat("SceneMerge: Merged %d base colors into averaged color: #%s",
				base_colors.size(), averaged_color.to_html(false)));
	} else {
		// Default to white if no valid materials found
		merged_material->set_albedo(Color(1, 1, 1, 1));
		print_line("SceneMerge: No valid BaseMaterial3D colors found, using default white");
	}

	// Apply merged material to all surfaces of the merged mesh
	for (int surface_i = 0; surface_i < merged_importer_mesh->get_surface_count(); surface_i++) {
		merged_importer_mesh->set_surface_material(surface_i, merged_material);
	}

	// Add merged mesh instance to scene
	ImporterMeshInstance3D *merged_instance = memnew(ImporterMeshInstance3D);
	merged_instance->set_mesh(merged_importer_mesh);
	merged_instance->set_name("MergedMesh");

	if (p_root != nullptr) {
		p_root->add_child(merged_instance);
		// Set owner to match scene ownership
		if (p_root->get_owner()) {
			merged_instance->set_owner(p_root->get_owner());
		}
	}

	return p_root;
}

void MeshTextureAtlas::_generate_texture_atlas(MergeState &state, String texture_type) {
#ifdef TOOLS_ENABLED
	EditorProgress progress_texture_atlas("gen_mesh_atlas", TTR("Generate Atlas"), state.atlas->meshCount);
	int step = 0;
#endif
	AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(state.atlas->width, state.atlas->height, false, Image::FORMAT_RGBA8);
	args.atlas_lookup = state.atlas_lookup.ptrw();
	args.atlas_height = state.atlas->height;
	args.atlas_width = state.atlas->width;
	for (uint32_t mesh_i = 0; mesh_i < state.atlas->meshCount; mesh_i++) {
		const xatlas::Mesh &mesh = state.atlas->meshes[mesh_i];
		for (uint32_t chart_i = 0; chart_i < mesh.chartCount; chart_i++) {
			const xatlas::Chart &chart = mesh.chartArray[chart_i];
			Ref<Image> img;
			if (texture_type == "albedo") {
				img = state.material_image_cache[chart.material].albedo_img;
			} else {
				ERR_PRINT("Unknown texture type: " + texture_type);
				continue;
			}
			ERR_CONTINUE(img.is_null());
			ERR_CONTINUE(img->is_empty());
			ERR_CONTINUE_MSG(Image::get_format_pixel_size(img->get_format()) > 4, "Float textures are not supported yet for texture type: " + texture_type);

			img->convert(Image::FORMAT_RGBA8);
			args.source_texture = img;
			args.material_index = (uint16_t)chart.material;

			for (uint32_t face_i = 0; face_i < chart.faceCount; face_i++) {
				Vector2 v[3];
				for (uint32_t l = 0; l < 3; l++) {
					const uint32_t index = mesh.indexArray[chart.faceArray[face_i] * 3 + l];
					const xatlas::Vertex &vertex = mesh.vertexArray[index];
					v[l] = Vector2(vertex.uv[0], vertex.uv[1]);
					args.source_uvs[l].x = state.uvs[mesh_i][vertex.xref].x / img->get_width();
					args.source_uvs[l].y = state.uvs[mesh_i][vertex.xref].y / img->get_height();
				}
				MeshMergeTriangle tri(v[0], v[1], v[2], Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1));

				tri.drawAA(set_atlas_texel, &args);
			}
		}
#ifdef TOOLS_ENABLED
		progress_texture_atlas.step(TTR("Process Mesh for Atlas: ") + texture_type + " (" + itos(step) + "/" + itos(state.atlas->meshCount) + ")", step);
		step++;
#endif
	}
	print_line(vformat("Generated atlas for %s: width=%d, height=%d", texture_type, args.atlas_data->get_width(), args.atlas_data->get_height()));
	args.atlas_data->generate_mipmaps();
	state.texture_atlas.insert(texture_type, args.atlas_data);
}

Ref<Image> MeshTextureAtlas::_get_source_texture(MergeState &state, Ref<BaseMaterial3D> material) {
	int32_t width = 0, height = 0;
	Vector<Ref<Texture2D> > textures = { material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO) };
	Vector<Ref<Image> > images;
	images.resize(textures.size());

	for (int i = 0; i < textures.size(); ++i) {
		if (textures[i].is_valid()) {
			images.write[i] = textures[i]->get_image();
			if (images[i].is_valid() && !images[i]->is_empty()) {
				width = MAX(width, images[i]->get_width());
				height = MAX(height, images[i]->get_height());
			}
		}
	}

	for (int i = 0; i < images.size(); ++i) {
		if (images[i].is_valid()) {
			if (!images[i]->is_empty() && images[i]->is_compressed()) {
				images.write[i]->decompress();
			}
			images.write[i]->resize(width, height, Image::INTERPOLATE_LANCZOS);
		}
	}

	Ref<Image> img = Image::create_empty(width, height, false, Image::FORMAT_RGBA8);

	bool has_albedo_texture = images[0].is_valid() && !images[0]->is_empty();
	Color color_mul = has_albedo_texture ? material->get_albedo() : Color(0, 0, 0, 0);
	Color color_add = has_albedo_texture ? Color(0, 0, 0, 0) : material->get_albedo();
	for (int32_t y = 0; y < img->get_height(); y++) {
		for (int32_t x = 0; x < img->get_width(); x++) {
			Color c = has_albedo_texture ? images[0]->get_pixel(x, y) : Color();
			c *= color_mul;
			c += color_add;
			img->set_pixel(x, y, c);
		}
	}

	return img;
}

Error MeshTextureAtlas::_generate_atlas(const int32_t p_num_meshes, Vector<Vector<Vector2> > &r_uvs, xatlas::Atlas *r_atlas, const Vector<MeshState> &r_meshes, const Vector<Ref<Material> > p_material_cache,
		xatlas::PackOptions &r_pack_options) {
	// Stub implementation - xatlas integration temporarily disabled
	// TODO: Re-enable when xatlas library issues are resolved
	return ERR_SKIP;
}

void MeshTextureAtlas::write_uvs(const Vector<MeshState> &p_mesh_items, Vector<Vector<Vector2> > &uv_groups, Array &r_mesh_to_index_to_material, Vector<Vector<ModelVertex> > &r_model_vertices) {
	int32_t total_surface_count = 0;
	for (int32_t mesh_i = 0; mesh_i < p_mesh_items.size(); mesh_i++) {
		total_surface_count += p_mesh_items[mesh_i].importer_mesh->get_surface_count();
	}
	r_model_vertices.resize(total_surface_count);
	uv_groups.resize(total_surface_count);

	int32_t mesh_count = 0;
	for (int32_t mesh_i = 0; mesh_i < p_mesh_items.size(); mesh_i++) {
		for (int32_t surface_i = 0; surface_i < p_mesh_items[mesh_i].importer_mesh->get_surface_count(); surface_i++) {
			Ref<ImporterMesh> importer_mesh = p_mesh_items[mesh_i].importer_mesh;
			Array mesh = importer_mesh->get_surface_arrays(surface_i);
			Vector<ModelVertex> model_vertices;
			Vector<Vector3> vertex_arr = mesh[Mesh::ARRAY_VERTEX];
			Vector<Vector3> normal_arr = mesh[Mesh::ARRAY_NORMAL];
			Vector<Vector2> uv_arr = mesh[Mesh::ARRAY_TEX_UV];
			Vector<int32_t> index_arr = mesh[Mesh::ARRAY_INDEX];
			Vector<Plane> tangent_arr = mesh[Mesh::ARRAY_TANGENT];
			Transform3D transform = p_mesh_items[mesh_i].importer_mesh_instance->get_transform();
			Node3D *parent_node = Node3D::cast_to<Node3D>(p_mesh_items[mesh_i].importer_mesh_instance->get_parent());
			for (; parent_node != nullptr; parent_node = Node3D::cast_to<Node3D>(parent_node->get_parent())) {
				transform = parent_node->get_transform() * transform;
			}
			if (!vertex_arr.is_empty()) {
				model_vertices.resize(vertex_arr.size());
			}

			model_vertices.resize(vertex_arr.size());
			Vector<Vector2> uvs;
			uvs.resize(vertex_arr.size());
			for (int32_t vertex_i = 0; vertex_i < vertex_arr.size(); vertex_i++) {
				ModelVertex vertex_attributes;
				vertex_attributes.pos = transform.xform(vertex_arr[vertex_i]);
				ERR_BREAK(normal_arr.is_empty());
				vertex_attributes.normal = normal_arr[vertex_i];
				vertex_attributes.normal.normalize();
				if (vertex_attributes.normal.length_squared() < CMP_EPSILON) {
					vertex_attributes.normal = Vector3(0, 1, 0);
				}
				model_vertices.write[vertex_i] = vertex_attributes;
				ERR_BREAK(r_mesh_to_index_to_material.is_empty());
				Array index_to_material = r_mesh_to_index_to_material[mesh_count];
				int32_t index = index_arr.find(vertex_i);
				ERR_CONTINUE(index == -1);

				uvs.write[vertex_i] = uv_arr[vertex_i];

				const Ref<Material> material = index_to_material.get(index);
				Ref<BaseMaterial3D> Node3D_material = material;
				const Ref<Texture2D> tex = Node3D_material->get_texture(BaseMaterial3D::TextureParam::TEXTURE_ALBEDO);
				if (tex.is_valid()) {
					uvs.write[vertex_i].x *= tex->get_width();
					uvs.write[vertex_i].y *= tex->get_height();
				}
			}
			r_model_vertices.write[mesh_count] = model_vertices;
			uv_groups.write[mesh_count] = uvs;
			mesh_count++;
		}
	}
}

Ref<Image> MeshTextureAtlas::dilate_image(Ref<Image> source_image) {
	Ref<Image> target_image = source_image->duplicate();
	target_image->convert(Image::FORMAT_RGBA8);
	LocalVector<uint8_t> pixels;
	int32_t height = target_image->get_size().y;
	int32_t width = target_image->get_size().x;
	const int32_t bytes_in_pixel = 4;
	pixels.resize(height * width * bytes_in_pixel);
	for (int32_t y = 0; y < height; y++) {
		for (int32_t x = 0; x < width; x++) {
			int32_t pixel_index = x + (width * y);
			int32_t index = pixel_index * bytes_in_pixel;
			Color pixel = target_image->get_pixel(x, y);
			pixels[index + 0] = uint8_t(pixel.r * 255.0f);
			pixels[index + 1] = uint8_t(pixel.g * 255.0f);
			pixels[index + 2] = uint8_t(pixel.b * 255.0f);
			pixels[index + 3] = uint8_t(pixel.a * 255.0f);
		}
	}
	rjm_texbleed(pixels.ptr(), width, height, 3, bytes_in_pixel, bytes_in_pixel * width);
	for (int32_t y = 0; y < height; y++) {
		for (int32_t x = 0; x < width; x++) {
			Color pixel;
			int32_t pixel_index = x + (width * y);
			int32_t index = bytes_in_pixel * pixel_index;
			pixel.r = pixels[index + 0] / 255.0f;
			pixel.g = pixels[index + 1] / 255.0f;
			pixel.b = pixels[index + 2] / 255.0f;
			pixel.a = 1.0f;
			target_image->set_pixel(x, y, pixel);
		}
	}
	target_image->generate_mipmaps();
	return target_image;
}

void MeshTextureAtlas::map_mesh_to_index_to_material(const Vector<MeshState> &p_mesh_items, Array &r_mesh_to_index_to_material, Vector<Ref<Material> > &r_material_cache) {
	float largest_dimension = 0;
	for (int32_t mesh_i = 0; mesh_i < p_mesh_items.size(); mesh_i++) {
		Ref<ImporterMesh> importer_mesh = p_mesh_items[mesh_i].importer_mesh;
		for (int32_t j = 0; j < importer_mesh->get_surface_count(); j++) {
			Ref<Material> mat = importer_mesh->get_surface_material(j);
			if (mat.is_null()) {
				continue;
			}
			Ref<BaseMaterial3D> base_mat = mat;
			if (base_mat.is_null()) {
				continue;
			}
			Ref<Texture2D> texture = base_mat->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
			if (texture.is_null()) {
				continue;
			}
			largest_dimension = MAX(texture->get_size().x, texture->get_size().y);
		}
	}
	for (int32_t mesh_i = 0; mesh_i < p_mesh_items.size(); mesh_i++) {
		Ref<ImporterMesh> importer_mesh = p_mesh_items[mesh_i].importer_mesh;
		// importer_mesh->mesh_unwrap(Transform3D(), TEXEL_SIZE); // Temporarily disabled

		for (int32_t j = 0; j < importer_mesh->get_surface_count(); j++) {
			Array mesh = importer_mesh->get_surface_arrays(j);
			Vector<Vector3> indices = mesh[Mesh::ARRAY_INDEX];
			Ref<Material> material = importer_mesh->get_surface_material(j);
			if (material.is_null()) {
				continue;
			}
			Ref<BaseMaterial3D> base_material = material;
			if (base_material.is_null() || base_material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO).is_null()) {
				Ref<Image> img = Image::create_empty(largest_dimension, largest_dimension, true, Image::FORMAT_RGBA8);
				img->fill(base_material.is_null() ? Color(1.0f, 1.0f, 1.0f) : base_material->get_albedo());
				if (base_material.is_valid()) {
					base_material->set_albedo(Color(1.0f, 1.0f, 1.0f));
				}
				Ref<ImageTexture> tex = ImageTexture::create_from_image(img);
				if (base_material.is_valid()) {
					base_material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, tex);
				}
			}
			if (r_material_cache.find(material) == -1) {
				r_material_cache.push_back(material);
			}
			Array materials;
			materials.resize(indices.size());
			for (int32_t index_i = 0; index_i < indices.size(); index_i++) {
				materials[index_i] = material;
			}
			r_mesh_to_index_to_material.push_back(materials);
		}
	}
}

Node *MeshTextureAtlas::_output_mesh_atlas(MergeState &state, int p_count) {
	if (state.atlas->width == 0 || state.atlas->height == 0) {
		return nullptr;
	}
	print_line(vformat("Atlas size: (%d, %d)", state.atlas->width, state.atlas->height));
	MeshTextureAtlas::TextureData texture_data;
	for (int32_t mesh_i = 0; mesh_i < state.r_mesh_items.size(); mesh_i++) {
		if (state.r_mesh_items[mesh_i].importer_mesh_instance->get_parent()) {
			Node3D *node_3d = memnew(Node3D);
			Transform3D transform = state.r_mesh_items[mesh_i].importer_mesh_instance->get_transform();
			node_3d->set_transform(transform);
			node_3d->set_name(state.r_mesh_items[mesh_i].importer_mesh_instance->get_name());
			state.r_mesh_items[mesh_i].importer_mesh_instance->replace_by(node_3d);
		}
	}
	Ref<SurfaceTool> surface_tool_all;
	surface_tool_all.instantiate();
	surface_tool_all->begin(Mesh::PRIMITIVE_TRIANGLES);
	for (uint32_t mesh_i = 0; mesh_i < state.atlas->meshCount; mesh_i++) {
		Ref<SurfaceTool> surface_tool;
		surface_tool.instantiate();
		surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
		const xatlas::Mesh &mesh = state.atlas->meshes[mesh_i];
		print_line(vformat("Mesh %d: vertexCount=%d, indexCount=%d", mesh_i, mesh.vertexCount, mesh.indexCount));
		uint32_t max_vertices = 32 * 1024;
		uint32_t num_parts = (mesh.vertexCount / max_vertices) + 1;
		print_line(vformat("Number of parts for Mesh %d: %d", mesh_i, num_parts));
		for (uint32_t part = 0; part < num_parts; part++) {
			uint32_t start = part * max_vertices;
			uint32_t end = MIN((part + 1) * max_vertices, mesh.vertexCount);
			print_line(vformat("Part %d: Start=%d, End=%d", part, start, end));

			for (uint32_t v = start; v < end; v++) {
				const xatlas::Vertex vertex = mesh.vertexArray[v];
				ERR_BREAK_MSG(vertex.xref < 0 || vertex.xref >= static_cast<uint32_t>(state.model_vertices[mesh_i].size()),
						"Vertex reference not found. " + vformat("Vertex %d: xref=%d", v, vertex.xref));
				const ModelVertex &sourceVertex = state.model_vertices[mesh_i][vertex.xref - start];
				Vector2 uv = Vector2(vertex.uv[0] / state.atlas->width, vertex.uv[1] / state.atlas->height);
				surface_tool->set_uv(uv);
				surface_tool->set_normal(sourceVertex.normal);
				surface_tool->set_color(Color(1.0f, 1.0f, 1.0f));
				surface_tool->add_vertex(sourceVertex.pos);
			}
			for (uint32_t i = 0; i < mesh.indexCount; i++) {
				uint32_t index = mesh.indexArray[i];
				surface_tool->add_index(index);
			}
			surface_tool->generate_tangents();
			Ref<ArrayMesh> array_mesh = surface_tool->commit();
			surface_tool_all->append_from(array_mesh, 0, Transform3D());
		}
	}
	Ref<StandardMaterial3D> material;
	material.instantiate();
	HashMap<String, Ref<Image> >::Iterator A = state.texture_atlas.find("albedo");
	if (A && !A->key.is_empty()) {
		Ref<Image> img = dilate_image(A->value);
		print_line(vformat("Albedo image size: (%d, %d)", img->get_width(), img->get_height()));
		Ref<ImageTexture> tex = ImageTexture::create_from_image(img);
		material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, tex);
	}
	material->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
	ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
	Ref<ArrayMesh> array_mesh = surface_tool_all->commit();
	Ref<ImporterMesh> importer_mesh = ImporterMesh::from_mesh(array_mesh);
	mesh_instance->set_mesh(importer_mesh);
	mesh_instance->set_name(state.p_name);
	Transform3D root_transform;
	mesh_instance->set_transform(root_transform.affine_inverse());
	return mesh_instance;
}

bool MeshTextureAtlas::MeshState::operator==(const MeshState &rhs) const {
	if (rhs.importer_mesh == importer_mesh && rhs.path == path && rhs.importer_mesh_instance == importer_mesh_instance) {
		return true;
	}
	return false;
}

Pair<int, int> MeshTextureAtlas::calculate_coordinates(const Vector2 &p_source_uv, int p_width, int p_height) {
	// Clamp UV coordinates to [0, 1] range to handle any out-of-bounds values
	float clamped_x = CLAMP(p_source_uv.x, 0.0f, 1.0f);
	float clamped_y = CLAMP(p_source_uv.y, 0.0f, 1.0f);

	// Convert to pixel coordinates, clamping to valid texture bounds
	int sx = CLAMP(static_cast<int>(round(clamped_x * (p_width - 1))), 0, p_width - 1);
	int sy = CLAMP(static_cast<int>(round(clamped_y * (p_height - 1))), 0, p_height - 1);

	return Pair<int, int>(sx, sy);
}

Vector2 MeshTextureAtlas::interpolate_source_uvs(const Vector3 &bar, const AtlasTextureArguments *args) {
	return args->source_uvs[0] * bar.x + args->source_uvs[1] * bar.y + args->source_uvs[2] * bar.z;
}

int MeshTextureAtlas::godot_xatlas_print(const char *p_print_string, ...) {
	if (is_print_verbose_enabled()) {
		va_list args;
		va_start(args, p_print_string);
		char formatted_string[1024];
		vsnprintf(formatted_string, sizeof(formatted_string), p_print_string, args);
		va_end(args);
		print_line_rich(String(formatted_string).strip_edges());
	}
	return OK;
}

bool MeshTextureAtlas::MeshState::is_valid() const {
	bool is_mesh_valid = importer_mesh.is_valid();
	if (!is_mesh_valid || importer_mesh_instance == nullptr) {
		return false;
	}
	int num_surfaces = importer_mesh->get_surface_count();
	for (int i = 0; i < num_surfaces; ++i) {
		Array arrays = importer_mesh->get_surface_arrays(i);
		Vector<Vector3> vertices = arrays[Mesh::ARRAY_VERTEX];
		Vector<int> indices = arrays[Mesh::ARRAY_INDEX];
		if (vertices.size() == 0 || indices.size() == 0) {
			return false;
		}
	}
	return true;
}

MeshTextureAtlas::MeshTextureAtlas() {
	xatlas::SetPrint(&godot_xatlas_print, true);
}
