/**************************************************************************/
/*  merge.h                                                               */
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

#pragma once

/*
MESH TEXTURE ATLAS MERGE SYSTEM

This module provides functionality for merging multiple 3D mesh instances into
a single optimized mesh with texture atlas generation.

EXTERNAL DEPENDENCIES:
- xAtlas (https://github.com/jpcy/xatlas) - UV unwrapping and atlas packing
- thekla_atlas (https://github.com/Thekla/thekla_atlas) - Alternative atlas methods
*/

#include "core/math/vector2.h"
#include "core/object/ref_counted.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/main/node.h"
#include "scene/resources/3d/importer_mesh.h"

#include "thirdparty/xatlas/xatlas.h"
#include <cstdint>

class MeshTextureAtlas {
public:
	// ========================================================================
	// CONFIGURATION CONSTANTS
	// ========================================================================

	/// Default texel size for UV unwrapping operations
	static constexpr float TEXEL_SIZE = 5.0f;

	// ========================================================================
	// CORE DATA STRUCTURES
	// ========================================================================

	/// Holds texture atlas image data and dimensions
	struct TextureData {
		uint16_t width; ///< Atlas width in pixels
		uint16_t height; ///< Atlas height in pixels
		int num_components; ///< Number of color components (RGB/RGBA)
		Ref<Image> image; ///< The atlas image data
	};

	/// Represents a single vertex in 3D space with normal and UV coordinates
	struct ModelVertex {
		Vector3 pos; ///< 3D position coordinates
		Vector3 normal; ///< Surface normal vector
		Vector2 uv; ///< UV texture coordinates
	};

	/// Describes the state of a mesh instance being processed for merging
	struct MeshState {
		Ref<ImporterMesh> importer_mesh; ///< The imported mesh data
		NodePath path; ///< Scene path to this mesh
		int32_t index_offset = 0; ///< Vertex offset for indexing
		ImporterMeshInstance3D *importer_mesh_instance; ///< The scene node instance

		/// Equality comparison for duplicate detection
		bool operator==(const MeshState &rhs) const;
		/// Validates that this mesh state contains valid mesh data
		bool is_valid() const;
	};

	/// Caches processed material textures to avoid redundant operations
	struct MaterialImageCache {
		Ref<Image> albedo_img; ///< Processed albedo texture
	};

	/// Represents a group of mesh states being merged together
	struct MeshMerge {
		Vector<MeshState> meshes; ///< Collection of mesh states to merge
		int vertex_count = 0; ///< Total vertices across all meshes
	};

	/// Overall state of the mesh merging process
	struct MeshMergeState {
		Vector<MeshMerge> mesh_items; ///< All mesh groups being processed
		Node *root = nullptr; ///< Root scene node
	};

	// ========================================================================
	// TEXTURE ATLAS STRUCTURES
	// ========================================================================

	/// Maps texture atlas pixels to source materials for lookup operations
	struct AtlasLookupTexel {
		uint16_t material_index = 0; ///< Index of the source material
		uint16_t x = 0; ///< Source texture X coordinate
		uint16_t y = 0; ///< Source texture Y coordinate
	};

	/// Arguments passed to atlas texel setting operations
	struct AtlasTextureArguments {
		Ref<Image> atlas_data; ///< The output atlas image
		Ref<Image> source_texture; ///< Source texture being sampled
		AtlasLookupTexel *atlas_lookup = nullptr; ///< Lookup table for material mapping
		uint16_t material_index = 0; ///< Current material being processed
		Vector2 source_uvs[3]; ///< UV coordinates for triangle vertices
		uint32_t atlas_width = 0; ///< Width of atlas in pixels
		uint32_t atlas_height = 0; ///< Height of atlas in pixels
	};

	// ========================================================================
	// MERGE PROCESSING STRUCTURES
	// ========================================================================

	/// Comprehensive state object for the entire merge process
	struct MergeState {
		Node *p_root = nullptr; ///< Scene root node
		xatlas::Atlas *atlas = nullptr; ///< xAtlas atlas data
		Vector<MeshState> &r_mesh_items; ///< Reference to mesh states
		Array &vertex_to_material; ///< Material mapping per vertex
		const Vector<Vector<Vector2> > uvs; ///< UV coordinate data
		const Vector<Vector<ModelVertex> > &model_vertices; ///< Processed vertex data
		String p_name; ///< Name for output node
		const xatlas::PackOptions &pack_options; ///< Atlas packing settings
		Vector<AtlasLookupTexel> &atlas_lookup; ///< Material lookup table
		Vector<Ref<Material> > &material_cache; ///< Cached materials
		HashMap<String, Ref<Image> > texture_atlas; ///< Generated atlas textures
		HashMap<int32_t, MaterialImageCache> material_image_cache; ///< Material image cache
	};

	// ========================================================================
	// PUBLIC API METHODS
	// ========================================================================

	/// Sets a single texel in the texture atlas during rasterization
	static bool set_atlas_texel(void *param, int x, int y, const Vector3 &bar,
			const Vector3 &dx, const Vector3 &dy, float coverage);

	/// Calculates pixel coordinates from UV coordinates and texture dimensions
	static Pair<int, int> calculate_coordinates(const Vector2 &sourceUv, int width, int height);

	/// Merges all mesh instances in a scene into a single optimized mesh
	static Node *merge_meshes(Node *p_root);

	/// Constructor - initializes xAtlas logging
	MeshTextureAtlas();

	/// Public wrapper for finding mesh instances (used by merge implementation)
	static void find_all_mesh_instances(Vector<MeshMerge> &r_items, Node *p_current_node, const Node *p_owner) {
		return _find_all_mesh_instances(r_items, p_current_node, p_owner);
	}

private:
	// ========================================================================
	// HELPER FUNCTIONS
	// ========================================================================

	/// Custom xAtlas logging function that outputs to Godot console
	static int godot_xatlas_print(const char *p_print_string, ...);

	/// Interpolates UV coordinates using barycentric weights
	static Vector2 interpolate_source_uvs(const Vector3 &bar, const AtlasTextureArguments *args);

	/// Expands atlas texture edges to prevent filtering artifacts
	static Ref<Image> dilate_image(Ref<Image> source_image);

	// ========================================================================
	// MESH DISCOVERY & PROCESSING
	// ========================================================================

	/// Recursively finds all ImporterMeshInstance3D nodes in the scene
	static void _find_all_mesh_instances(Vector<MeshMerge> &r_items, Node *p_current_node, const Node *p_owner);

	/// Processes UV coordinates for mesh unwrapping
	static void write_uvs(const Vector<MeshState> &p_mesh_items, Vector<Vector<Vector2> > &uv_groups,
			Array &r_vertex_to_material, Vector<Vector<ModelVertex> > &r_model_vertices);

	/// Maps mesh vertices to their corresponding materials
	static void map_mesh_to_index_to_material(const Vector<MeshState> &mesh_items, Array &vertex_to_material,
			Vector<Ref<Material> > &material_cache);

	// ========================================================================
	// TEXTURE ATLAS GENERATION
	// ========================================================================

	/// Generates texture atlas for a specific material type (albedo, normal, etc.)
	static void _generate_texture_atlas(MergeState &state, String texture_type);

	/// Retrieves or processes source texture for a given material
	static Ref<Image> _get_source_texture(MergeState &state, Ref<BaseMaterial3D> material);

	/// Main atlas generation function using xAtlas library
	static Error _generate_atlas(const int32_t p_num_meshes, Vector<Vector<Vector2> > &r_uvs,
			xatlas::Atlas *atlas, const Vector<MeshState> &r_meshes,
			const Vector<Ref<Material> > material_cache, xatlas::PackOptions &pack_options);

	/// Creates the final merged mesh node with atlas materials
	static Node *_output_mesh_atlas(MergeState &state, int p_count);

protected:
	/// Registers methods with Godot's scripting system
	static void _bind_methods();
};
