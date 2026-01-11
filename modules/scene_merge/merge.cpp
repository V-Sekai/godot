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

			mesh.vertex_count += PackedVector3Array(array[Mesh::ARRAY_VERTEX]).size();
			mesh_state.index_offset = mesh.vertex_count;

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

Node *MeshTextureAtlas::merge_meshes(Node *p_root) {
	// Simple MVP: Merge all ImporterMeshInstance3D nodes into a single mesh
	// without texture atlas generation

	Vector<MeshMerge> mesh_merges;
	MeshMerge mesh_merge;
	mesh_merges.push_back(mesh_merge);

	// Find all ImporterMeshInstance3D nodes in the scene
	_find_all_mesh_instances(mesh_merges, p_root, nullptr);

	if (mesh_merges[0].meshes.size() <= 1) {
		// Nothing to merge
		return p_root;
	}

	// Create a surface tool to build the merged mesh
	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);

	// Process each mesh state
	for (const MeshState &mesh_state : mesh_merges[0].meshes) {
		Ref<ImporterMesh> mesh = mesh_state.importer_mesh;
		ImporterMeshInstance3D *mi = mesh_state.importer_mesh_instance;
		if (mesh.is_null()) {
			continue;
		}

		Transform3D transform = mi->get_global_transform();

		// Add each surface of the mesh
		for (int surface_i = 0; surface_i < mesh->get_surface_count(); surface_i++) {
			Array surface_arrays = mesh->get_surface_arrays(surface_i);
			Ref<Material> material = mesh->get_surface_material(surface_i);

			// Get vertex data
			Vector<Vector3> vertices = surface_arrays[Mesh::ARRAY_VERTEX];
			Vector<Vector3> normals = surface_arrays[Mesh::ARRAY_NORMAL];
			Vector<int> indices = surface_arrays[Mesh::ARRAY_INDEX];

			// Transform vertices and normals
			for (int i = 0; i < vertices.size(); i++) {
				vertices.write[i] = transform.xform(vertices[i]);
				if (i < normals.size()) {
					normals.write[i] = transform.basis.xform(normals[i]);
				}
			}

			// Add to surface tool
			st->set_material(material);
			for (int i = 0; i < vertices.size(); i++) {
				if (i < normals.size()) {
					st->set_normal(normals[i]);
				}
				st->add_vertex(vertices[i]);
			}

			// Add indices if present
			if (!indices.is_empty()) {
				for (int i = 0; i < indices.size(); i += 3) {
					st->add_index(indices[i]);
					st->add_index(indices[i + 1]);
					st->add_index(indices[i + 2]);
				}
			}
		}

		// Remove the original mesh instance
		Node *parent = mi->get_parent();
		if (parent) {
			parent->remove_child(mi);
			mi->queue_free();
		}
	}

	// Create the merged mesh
	Ref<ArrayMesh> merged_mesh;
	merged_mesh.instantiate();
	st->generate_normals();
	st->commit(merged_mesh);

	// Create a new ImporterMeshInstance3D with the merged mesh
	ImporterMeshInstance3D *merged_instance = memnew(ImporterMeshInstance3D);
	Ref<ImporterMesh> importer_mesh = ImporterMesh::from_mesh(merged_mesh);
	merged_instance->set_mesh(importer_mesh);
	merged_instance->set_name("MergedMesh");

	// Add it to the scene (attach to root or first found parent)
	p_root->add_child(merged_instance);

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
	int sx, sy;
	sx = static_cast<int>(round(p_source_uv.x * p_width)) % p_width;
	sy = static_cast<int>(round(p_source_uv.y * p_height)) % p_height;
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
