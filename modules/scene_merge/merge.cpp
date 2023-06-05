/*************************************************************************/
/*  mesh.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

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

#include "core/core_bind.h"
#include "core/io/image.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/os/os.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "scene/3d/node_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/mesh_data_tool.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/surface_tool.h"

#include "thirdparty/misc/rjm_texbleed.h"
#include "thirdparty/xatlas/xatlas.h"
#include <time.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "merge.h"

void SceneMerge::merge(const String p_file, Node *p_root_node) {
	PackedScene *scene = memnew(PackedScene);
	scene->pack(p_root_node);
	Node *root = scene->instantiate();
	Ref<MeshMergeMaterialRepack> repack;
	repack.instantiate();
	root = repack->merge(root, p_root_node, p_file);
	ERR_FAIL_COND(!root);
	scene->pack(root);
	ResourceSaver::save(scene, p_file);
}

bool MeshMergeMaterialRepack::setAtlasTexel(void *param, int x, int y, const Vector3 &bar, const Vector3 &, const Vector3 &, float) {
    SetAtlasTexelArgs *args = static_cast<SetAtlasTexelArgs *>(param);
    if (args->sourceTexture.is_valid()) {
        // Interpolate source UVs using barycentrics.
        const Vector2 sourceUv = args->source_uvs[0] * bar.x + args->source_uvs[1] * bar.y + args->source_uvs[2] * bar.z;
        
        // Keep coordinates in range of texture dimensions.
        int _width = args->sourceTexture->get_width() - 1;
        int _height = args->sourceTexture->get_height() - 1;

        int sx = static_cast<int>(sourceUv.x * _width) % _width;
        int sy = static_cast<int>(sourceUv.y * _height) % _height;

        if (sx < 0) {
            sx += _width;
        }
        if (sy < 0) {
            sy += _height;
        }

        const Color color = args->sourceTexture->get_pixel(sx, sy);
        args->atlasData->set_pixel(x, y, color);

        AtlasLookupTexel &lookup = args->atlas_lookup[x * y + args->atlas_width];
        lookup.material_index = args->material_index;
        lookup.x = static_cast<uint16_t>(sx);
        lookup.y = static_cast<uint16_t>(sy);

        return true;
    }
    return false;
}

void MeshMergeMaterialRepack::_find_all_mesh_instances(Vector<MeshMerge> &r_items, Node *p_current_node, const Node *p_owner) {
	MeshInstance3D *mi = cast_to<MeshInstance3D>(p_current_node);
	if (mi && mi->is_visible() && mi->get_mesh().is_valid()) {
		Ref<Mesh> array_mesh = mi->get_mesh();
		bool has_blends = false, has_bones = false, has_transparency = false;

		for (int32_t surface_i = 0; surface_i < array_mesh->get_surface_count(); surface_i++) {
			Ref<Material> active_material = mi->get_active_material(surface_i);
			if (active_material.is_null()) {
				active_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
			}
			array_mesh->surface_set_material(surface_i, active_material);

			Array array = array_mesh->surface_get_arrays(surface_i).duplicate(true);
			has_bones |= PackedFloat32Array(array[ArrayMesh::ARRAY_BONES]).size() != 0;
			has_blends |= array_mesh->get_blend_shape_count() != 0;
			Ref<BaseMaterial3D> base_mat = array_mesh->surface_get_material(surface_i);
			if (base_mat.is_valid()) {
				has_transparency |= base_mat->get_transparency() != BaseMaterial3D::TRANSPARENCY_DISABLED;
			}

			if (has_blends || has_bones || has_transparency) {
				break;
			}

			MeshState mesh_state;
			mesh_state.mesh = array_mesh;
			if (mi->is_inside_tree()) {
				mesh_state.path = mi->get_path();
			}
			mesh_state.mesh_instance = mi;
			MeshMerge &mesh = r_items.write[r_items.size() - 1];
			mesh.vertex_count += PackedVector3Array(array[ArrayMesh::ARRAY_VERTEX]).size();
			mesh.meshes.push_back(mesh_state);
		}
	}

	for (int32_t child_i = 0; child_i < p_current_node->get_child_count(); child_i++) {
		_find_all_mesh_instances(r_items, p_current_node->get_child(child_i), p_owner);
	}
}

void MeshMergeMaterialRepack::_bind_methods() {
	ClassDB::bind_method(D_METHOD("merge", "root", "original_root", "output_path"), &MeshMergeMaterialRepack::merge);
}

Node *MeshMergeMaterialRepack::merge(Node *p_root, Node *p_original_root, String p_output_path) {
	MeshMergeState mesh_merge_state;
	mesh_merge_state.root = p_root;
	mesh_merge_state.original_root = p_original_root;
	mesh_merge_state.output_path = p_output_path;
	mesh_merge_state.mesh_items.resize(1);
	_find_all_mesh_instances(mesh_merge_state.mesh_items, p_root, p_root);

	mesh_merge_state.original_mesh_items.resize(1);
	_find_all_mesh_instances(mesh_merge_state.original_mesh_items, p_original_root, p_original_root);
	if (mesh_merge_state.original_mesh_items.size() != mesh_merge_state.mesh_items.size()) {
		return p_root;
	}

	for (int32_t items_i = 0; items_i < mesh_merge_state.mesh_items.size(); items_i++) {
		p_root = _merge_list(mesh_merge_state, items_i);
	}
	return p_root;
}

Node *MeshMergeMaterialRepack::_merge_list(MeshMergeState p_mesh_merge_state, int p_index) {
	Vector<MeshState> mesh_items = p_mesh_merge_state.mesh_items[p_index].meshes;
	Node *p_root = p_mesh_merge_state.root;
	const Vector<MeshState> &original_mesh_items = p_mesh_merge_state.original_mesh_items[p_index].meshes;
	Array mesh_to_index_to_material;
	Vector<Ref<Material> > material_cache;

	map_mesh_to_index_to_material(mesh_items, mesh_to_index_to_material, material_cache);

	Vector<Vector<Vector2> > uv_groups;
	Vector<Vector<ModelVertex> > model_vertices;
	scale_uvs_by_texture_dimension_larger(original_mesh_items, mesh_items, uv_groups, mesh_to_index_to_material, model_vertices);
	xatlas::Atlas *atlas = xatlas::Create();

	int32_t num_surfaces = 0;
	for (const MeshState &mesh_item : mesh_items) {
		num_surfaces += mesh_item.mesh->get_surface_count();
	}

	xatlas::PackOptions pack_options;
	Vector<AtlasLookupTexel> atlas_lookup;
	_generate_atlas(num_surfaces, uv_groups, atlas, mesh_items, material_cache, pack_options);
	atlas_lookup.resize(atlas->width * atlas->height);
	HashMap<String, Ref<Image> > texture_atlas;
	HashMap<int32_t, MaterialImageCache> material_image_cache;

	MergeState state{
		p_root,
		atlas,
		mesh_items,
		mesh_to_index_to_material,
		uv_groups,
		model_vertices,
		p_root->get_name(),
		p_mesh_merge_state.output_path,
		pack_options,
		atlas_lookup,
		material_cache,
		texture_atlas,
		material_image_cache,
	};

#ifdef TOOLS_ENABLED
	EditorProgress progress_scene_merge("gen_get_source_material", TTR("Get source material"), state.material_cache.size());
	int step = 0;
#endif

	for (const Ref<Material> &abstract_material : state.material_cache) {
#ifdef TOOLS_ENABLED
		step++;
#endif
		Ref<BaseMaterial3D> material = abstract_material;
		MaterialImageCache cache{
			_get_source_texture(state, material),
			Ref<Image>(),
			Ref<Image>(),
			Ref<Image>(),
		};
		int32_t material_i = state.material_cache.find(abstract_material);
		state.material_image_cache[material_i == -1 ? state.material_image_cache.size() : material_i] = cache;

#ifdef TOOLS_ENABLED
		progress_scene_merge.step(TTR("Getting Source Material: ") + material->get_name() + " (" + itos(step) + "/" + itos(state.material_cache.size()) + ")", step);
#endif
	}

	_generate_texture_atlas(state, "albedo");

	if (state.atlas->width <= 0 && state.atlas->height <= 0) {
		xatlas::Destroy(atlas);
		return state.p_root;
	}

	p_root = _output(state, p_index);

	xatlas::Destroy(atlas);
	return p_root;
}

void MeshMergeMaterialRepack::_generate_texture_atlas(MergeState &state, String texture_type) {
	Ref<Image> atlas_img = Image::create_empty(state.atlas->width, state.atlas->height, false, Image::FORMAT_RGBA8);
	ERR_FAIL_COND_MSG(atlas_img.is_null(), "Failed to create empty atlas image.");

	// Rasterize chart triangles.
#ifdef TOOLS_ENABLED
	EditorProgress progress_texture_atlas("gen_mesh_atlas", TTR("Generate Atlas"), state.atlas->meshCount);
	int step = 0;
#endif
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
			SetAtlasTexelArgs args;
			args.sourceTexture = img;
			args.atlasData = atlas_img;
			args.atlas_lookup = state.atlas_lookup.ptrw();
			args.atlas_height = state.atlas->height;
			args.atlas_width = state.atlas->width;
			args.material_index = (uint16_t)chart.material;
			for (uint32_t face_i = 0; face_i < chart.faceCount; face_i++) {
				Vector2 v[3];
				for (uint32_t l = 0; l < 3; l++) {
					const uint32_t index = mesh.indexArray[chart.faceArray[face_i] * 3 + l];
					const xatlas::Vertex &vertex = mesh.vertexArray[index];
					v[l] = Vector2(vertex.uv[0], vertex.uv[1]);
					int img_width = img->get_width();
					int img_height = img->get_height();
					ERR_CONTINUE_MSG(img_width == 0 || img_height == 0, "Image width or height is zero for texture type: " + texture_type);

					args.source_uvs[l].x = state.uvs[mesh_i][vertex.xref].x / img_width;
					args.source_uvs[l].y = state.uvs[mesh_i][vertex.xref].y / img_height;
				}
				Triangle tri(v[0], v[1], v[2], Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1));

				tri.drawAA(setAtlasTexel, &args);
			}
		}
#ifdef TOOLS_ENABLED
		progress_texture_atlas.step(TTR("Process Mesh for Atlas: ") + texture_type + " (" + itos(step) + "/" + itos(state.atlas->meshCount) + ")", step);
		step++;
#endif
	}
	if (atlas_img.is_valid()) {
		print_line(vformat("Generated atlas for %s: width=%d, height=%d", texture_type, atlas_img->get_width(), atlas_img->get_height()));
	} else {
		print_line(vformat("Failed to generate atlas for %s", texture_type));
	}
	if (!atlas_img->is_empty()) {
		atlas_img->generate_mipmaps();
	}
	state.texture_atlas.insert(texture_type, atlas_img);
}

Ref<Image> MeshMergeMaterialRepack::_get_source_texture(MergeState &state, Ref<BaseMaterial3D> material) {
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

void MeshMergeMaterialRepack::_generate_atlas(const int32_t p_num_meshes, Vector<Vector<Vector2> > &r_uvs, xatlas::Atlas *atlas, const Vector<MeshState> &r_meshes, const Vector<Ref<Material> > material_cache,
		xatlas::PackOptions &pack_options) {
	uint32_t mesh_count = 0;
	for (int32_t mesh_i = 0; mesh_i < r_meshes.size(); mesh_i++) {
		for (int32_t j = 0; j < r_meshes[mesh_i].mesh->get_surface_count(); j++) {
			Array mesh = r_meshes[mesh_i].mesh->surface_get_arrays(j);
			if (mesh.is_empty()) {
				xatlas::UvMeshDecl meshDecl;
				xatlas::AddUvMesh(atlas, meshDecl);
				mesh_count++;
				continue;
			}
			Array indices = mesh[ArrayMesh::ARRAY_INDEX];
			if (!indices.size()) {
				xatlas::UvMeshDecl meshDecl;
				xatlas::AddUvMesh(atlas, meshDecl);
				mesh_count++;
				continue;
			}
			xatlas::UvMeshDecl meshDecl;
			meshDecl.vertexCount = r_uvs[mesh_count].size();
			meshDecl.vertexUvData = r_uvs[mesh_count].ptr();
			meshDecl.vertexStride = sizeof(Vector2);
			Vector<int32_t> mesh_indices = mesh[Mesh::ARRAY_INDEX];
			Vector<uint32_t> indexes;
			indexes.resize(mesh_indices.size());
			Vector<uint32_t> materials;
			materials.resize(mesh_indices.size());
			for (int32_t index_i = 0; index_i < mesh_indices.size(); index_i++) {
				indexes.write[index_i] = mesh_indices[index_i];
			}
			for (int32_t index_i = 0; index_i < mesh_indices.size(); index_i++) {
				Ref<Material> mat = r_meshes[mesh_i].mesh->surface_get_material(j);
				int32_t material_i = material_cache.find(mat);
				if (material_i < 0 || material_i >= material_cache.size()) {
					continue;
				}
				if (material_i != -1) {
					materials.write[index_i] = material_i;
				}
			}
			meshDecl.indexCount = indexes.size();
			meshDecl.indexData = indexes.ptr();
			meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
			meshDecl.faceMaterialData = materials.ptr();
			xatlas::AddMeshError error = xatlas::AddUvMesh(atlas, meshDecl);
			ERR_CONTINUE_MSG(error != xatlas::AddMeshError::Success, String("Error adding mesh ") + itos(mesh_i) + String(": ") + xatlas::StringForEnum(error));
			mesh_count++;
		}
	}
	pack_options.bilinear = true;
	pack_options.padding = 16;
	pack_options.texelsPerUnit = 0.0f;
	pack_options.bruteForce = false;
	pack_options.blockAlign = true;
	pack_options.resolution = 2048;
	xatlas::ComputeCharts(atlas);
	xatlas::PackCharts(atlas, pack_options);
}

void MeshMergeMaterialRepack::scale_uvs_by_texture_dimension_larger(const Vector<MeshState> &original_mesh_items, Vector<MeshState> &mesh_items, Vector<Vector<Vector2> > &uv_groups, Array &r_mesh_to_index_to_material, Vector<Vector<ModelVertex> > &r_model_vertices) {
	int32_t total_surface_count = 0;
	for (int32_t mesh_i = 0; mesh_i < mesh_items.size(); mesh_i++) {
		total_surface_count += mesh_items[mesh_i].mesh->get_surface_count();
	}

	r_model_vertices.resize(total_surface_count);
	uv_groups.resize(total_surface_count);

	int32_t mesh_count = 0;
	for (int32_t mesh_i = 0; mesh_i < mesh_items.size(); mesh_i++) {
		for (int32_t surface_i = 0; surface_i < mesh_items[mesh_i].mesh->get_surface_count(); surface_i++) {
			Ref<ArrayMesh> array_mesh = mesh_items[mesh_i].mesh;
			Array mesh = array_mesh->surface_get_arrays(surface_i);
			Vector<ModelVertex> model_vertices;
			if (mesh.is_empty()) {
				mesh_count++;
				r_model_vertices.write[mesh_count] = model_vertices;
				continue;
			}
			Array vertices = mesh[ArrayMesh::ARRAY_VERTEX];
			if (vertices.size() == 0) {
				mesh_count++;
				r_model_vertices.write[mesh_count] = model_vertices;
				continue;
			}
			Vector<Vector3> vertex_arr = mesh[Mesh::ARRAY_VERTEX];
			Vector<Vector3> normal_arr = mesh[Mesh::ARRAY_NORMAL];
			Vector<Vector2> uv_arr = mesh[Mesh::ARRAY_TEX_UV];
			Vector<int32_t> index_arr = mesh[Mesh::ARRAY_INDEX];
			Vector<Plane> tangent_arr = mesh[Mesh::ARRAY_TANGENT];
			Transform3D xform = original_mesh_items[mesh_i].mesh_instance->get_global_transform();
			model_vertices.resize(vertex_arr.size());
			for (int32_t vertex_i = 0; vertex_i < vertex_arr.size(); vertex_i++) {
				ModelVertex vertex;
				vertex.pos = xform.xform(vertex_arr[vertex_i]);
				if (uv_arr.size()) {
					vertex.uv = uv_arr[vertex_i];
				}
				if (normal_arr.size()) {
					vertex.normal = normal_arr[vertex_i];
				}
				model_vertices.write[vertex_i] = vertex;
			}
			r_model_vertices.write[mesh_count] = model_vertices;
			mesh_count++;
		}
	}

	mesh_count = 0;
	for (int32_t mesh_i = 0; mesh_i < mesh_items.size(); mesh_i++) {
		for (int32_t j = 0; j < mesh_items[mesh_i].mesh->get_surface_count(); j++) {
			Array mesh = mesh_items[mesh_i].mesh->surface_get_arrays(j);
			if (mesh.is_empty()) {
				uv_groups.push_back(Vector<Vector2>());
				mesh_count++;
				continue;
			}
			Vector<Vector3> vertices = mesh[ArrayMesh::ARRAY_VERTEX];
			if (vertices.size() == 0) {
				mesh_count++;
				uv_groups.push_back(Vector<Vector2>());
				continue;
			}
			Vector<Vector2> uvs;
			uvs.resize(vertices.size());
			Vector<int32_t> indices = mesh[ArrayMesh::ARRAY_INDEX];
			for (int32_t vertex_i = 0; vertex_i < vertices.size(); vertex_i++) {
				if (mesh_count >= r_mesh_to_index_to_material.size()) {
					uvs.resize(0);
					break;
				}
				Array index_to_material = r_mesh_to_index_to_material[mesh_count];
				if (!index_to_material.size()) {
					continue;
				}
				int32_t index = indices.find(vertex_i);
				if (index >= index_to_material.size()) {
					continue;
				}
				ERR_CONTINUE(index == -1);
				const Ref<Material> material = index_to_material.get(index);
				Ref<BaseMaterial3D> Node3D_material = material;
				const Ref<Texture2D> tex = Node3D_material->get_texture(BaseMaterial3D::TextureParam::TEXTURE_ALBEDO);
				uvs.write[vertex_i] = r_model_vertices[mesh_count][vertex_i].uv;
				if (tex.is_valid()) {
					uvs.write[vertex_i].x *= tex->get_width();
					uvs.write[vertex_i].y *= tex->get_height();
				}
			}
			uv_groups.write[mesh_count] = uvs;
			mesh_count++;
		}
	}
}

Ref<Image> MeshMergeMaterialRepack::dilate(Ref<Image> source_image) {
	Ref<Image> target_image = source_image->duplicate();
	target_image->convert(Image::FORMAT_RGBA8);
	Vector<uint8_t> pixels;
	int32_t height = target_image->get_size().y;
	int32_t width = target_image->get_size().x;
	const int32_t bytes_in_pixel = 4;
	pixels.resize(height * width * bytes_in_pixel);
	for (int32_t y = 0; y < height; y++) {
		for (int32_t x = 0; x < width; x++) {
			int32_t pixel_index = x + (width * y);
			int32_t index = pixel_index * bytes_in_pixel;
			Color pixel = target_image->get_pixel(x, y);
			pixels.write[index + 0] = uint8_t(pixel.r * 255.0f);
			pixels.write[index + 1] = uint8_t(pixel.g * 255.0f);
			pixels.write[index + 2] = uint8_t(pixel.b * 255.0f);
			pixels.write[index + 3] = uint8_t(pixel.a * 255.0f);
		}
	}
	rjm_texbleed(pixels.ptrw(), width, height, 3, bytes_in_pixel, bytes_in_pixel * width);
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

void MeshMergeMaterialRepack::map_mesh_to_index_to_material(Vector<MeshState> mesh_items, Array &mesh_to_index_to_material, Vector<Ref<Material> > &material_cache) {
	float largest_dimension = 0;
	for (int32_t mesh_i = 0; mesh_i < mesh_items.size(); mesh_i++) {
		Ref<ArrayMesh> array_mesh = mesh_items[mesh_i].mesh;
		for (int32_t j = 0; j < array_mesh->get_surface_count(); j++) {
			Ref<BaseMaterial3D> mat = array_mesh->surface_get_material(j);
			Ref<Texture2D> texture = mat->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
			if (texture.is_valid()) {
				largest_dimension = MAX(texture->get_size().x, texture->get_size().y);
			}
		}
	}
	largest_dimension = MAX(largest_dimension, default_texture_length);
	for (int32_t mesh_i = 0; mesh_i < mesh_items.size(); mesh_i++) {
		Ref<ArrayMesh> array_mesh = mesh_items[mesh_i].mesh;
		array_mesh->lightmap_unwrap(Transform3D(), 1.0f / largest_dimension, true);
		for (int32_t j = 0; j < array_mesh->get_surface_count(); j++) {
			Array mesh = array_mesh->surface_get_arrays(j);
			Vector<Vector3> indices = mesh[ArrayMesh::ARRAY_INDEX];
			Ref<BaseMaterial3D> material = mesh_items[mesh_i].mesh->surface_get_material(j);
			if (material->get_texture(BaseMaterial3D::TEXTURE_ALBEDO).is_null()) {
				Ref<Image> img = Image::create_empty(default_texture_length, default_texture_length, true, Image::FORMAT_RGBA8);
				img->fill(material->get_albedo());
				material->set_albedo(Color(1.0f, 1.0f, 1.0f));
				Ref<ImageTexture> tex = ImageTexture::create_from_image(img);
				material->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, tex);
			}

			if (material_cache.find(material) == -1) {
				material_cache.push_back(material);
			}
			Array materials;
			materials.resize(indices.size());
			for (int32_t index_i = 0; index_i < indices.size(); index_i++) {
				materials[index_i] = material;
			}
			mesh_to_index_to_material.push_back(materials);
		}
	}
}

Node *MeshMergeMaterialRepack::_output(MergeState &state, int p_count) {
	if (state.atlas->width == 0 || state.atlas->height == 0) {
		return state.p_root;
	}
	print_line(vformat("Atlas size: (%d, %d)", state.atlas->width, state.atlas->height));
	MeshMergeMaterialRepack::TextureData texture_data;
	for (int32_t mesh_i = 0; mesh_i < state.r_mesh_items.size(); mesh_i++) {
		if (state.r_mesh_items[mesh_i].mesh_instance->get_parent()) {
			Node3D *node_3d = memnew(Node3D);
			Transform3D xform = state.r_mesh_items[mesh_i].mesh_instance->get_transform();
			node_3d->set_transform(xform);
			node_3d->set_name(state.r_mesh_items[mesh_i].mesh_instance->get_name());
			state.r_mesh_items[mesh_i].mesh_instance->replace_by(node_3d);
		}
	}
	Ref<SurfaceTool> st_all;
	st_all.instantiate();
	st_all->begin(Mesh::PRIMITIVE_TRIANGLES);
	for (uint32_t mesh_i = 0; mesh_i < state.atlas->meshCount; mesh_i++) {
		Ref<SurfaceTool> st;
		st.instantiate();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);
		const xatlas::Mesh &mesh = state.atlas->meshes[mesh_i];
		print_line(vformat("Mesh %d: vertexCount=%d, indexCount=%d", mesh_i, mesh.vertexCount, mesh.indexCount));
		for (uint32_t v = 0; v < mesh.vertexCount; v++) {
			const xatlas::Vertex vertex = mesh.vertexArray[v];
			const ModelVertex &sourceVertex = state.model_vertices[mesh_i][vertex.xref];
			Vector2 uv = Vector2(vertex.uv[0] / state.atlas->width, vertex.uv[1] / state.atlas->height);
			st->set_uv(uv);
			st->set_normal(sourceVertex.normal);
			st->set_color(Color(1.0f, 1.0f, 1.0f));
			st->add_vertex(sourceVertex.pos);
		}
		for (uint32_t f = 0; f < mesh.indexCount; f++) {
			const uint32_t index = mesh.indexArray[f];
			st->add_index(index);
		}
		st->generate_tangents();
		Ref<ArrayMesh> array_mesh = st->commit();
		st_all->append_from(array_mesh, 0, Transform3D());
	}
	Ref<StandardMaterial3D> mat;
	mat.instantiate();
	mat->set_name("Atlas");
	HashMap<String, Ref<Image> >::Iterator A = state.texture_atlas.find("albedo");
	Image::CompressMode compress_mode = Image::COMPRESS_ETC;
	if (Image::_image_compress_bc_func) {
		compress_mode = Image::COMPRESS_S3TC;
	}
	if (A && !A->key.is_empty()) {
		Ref<Image> img = dilate(A->value);
		print_line(vformat("Albedo image size: (%d, %d)", img->get_width(), img->get_height()));
		img->compress(compress_mode, Image::COMPRESS_SOURCE_SRGB);
		String path = state.output_path;
		String base_dir = path.get_base_dir();
		path = base_dir.path_to_file(path.get_basename().get_file() + "_albedo");
		Ref<DirAccess> directory = DirAccess::create(DirAccess::AccessType::ACCESS_FILESYSTEM);
		path += "_" + itos(p_count) + ".res";
		Ref<ImageTexture> tex = ImageTexture::create_from_image(img);
		ResourceSaver::save(tex, path);
		Ref<Texture2D> res = ResourceLoader::load(path, "Texture2D");
		mat->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, res);
	}
	mat->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
	MeshInstance3D *mi = memnew(MeshInstance3D);
	Ref<ArrayMesh> array_mesh = st_all->commit();
	mi->set_mesh(array_mesh);
	mi->set_name(state.p_name);
	Transform3D root_xform;
	Node3D *node_3d = cast_to<Node3D>(state.p_root);
	if (node_3d) {
		root_xform = node_3d->get_transform();
	}
	mi->set_transform(root_xform.affine_inverse());
	array_mesh->surface_set_material(0, mat);
	state.p_root->add_child(mi, true);
	if (mi != state.p_root) {
		mi->set_owner(state.p_root);
	}
	return state.p_root;
}

bool MeshMergeMaterialRepack::MeshState::operator==(const MeshState &rhs) const {
	if (rhs.mesh == mesh && rhs.path == path && rhs.mesh_instance == mesh_instance) {
		return true;
	}
	return false;
}

MeshMergeMaterialRepack::ClippedTriangle::ClippedTriangle(const Vector2 &a, const Vector2 &b, const Vector2 &c) {
	m_area = 0;
	m_numVertices = 3;
	m_activeVertexBuffer = 0;
	m_verticesA[0] = a;
	m_verticesA[1] = b;
	m_verticesA[2] = c;
	m_vertexBuffers[0] = m_verticesA;
	m_vertexBuffers[1] = m_verticesB;
}

void MeshMergeMaterialRepack::ClippedTriangle::clipHorizontalPlane(float offset, float clipdirection) {
	Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
	m_activeVertexBuffer ^= 1;
	Vector2 *v2 = m_vertexBuffers[m_activeVertexBuffer];
	v[m_numVertices] = v[0];
	float dy2, dy1 = offset - v[0].y;
	int dy2in, dy1in = clipdirection * dy1 >= 0;
	uint32_t p = 0;
	for (uint32_t k = 0; k < m_numVertices; k++) {
		dy2 = offset - v[k + 1].y;
		dy2in = clipdirection * dy2 >= 0;
		if (dy1in) {
			v2[p++] = v[k];
		}
		if (dy1in + dy2in == 1) { // not both in/out
			float dx = v[k + 1].x - v[k].x;
			float dy = v[k + 1].y - v[k].y;
			v2[p++] = Vector2(v[k].x + dy1 * (dx / dy), offset);
		}
		dy1 = dy2;
		dy1in = dy2in;
	}
	m_numVertices = p;
}

void MeshMergeMaterialRepack::ClippedTriangle::clipVerticalPlane(float offset, float clipdirection) {
	Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
	m_activeVertexBuffer ^= 1;
	Vector2 *v2 = m_vertexBuffers[m_activeVertexBuffer];
	v[m_numVertices] = v[0];
	float dx2, dx1 = offset - v[0].x;
	int dx2in, dx1in = clipdirection * dx1 >= 0;
	uint32_t p = 0;
	for (uint32_t k = 0; k < m_numVertices; k++) {
		dx2 = offset - v[k + 1].x;
		dx2in = clipdirection * dx2 >= 0;
		if (dx1in) {
			v2[p++] = v[k];
		}
		if (dx1in + dx2in == 1) { // not both in/out
			float dx = v[k + 1].x - v[k].x;
			float dy = v[k + 1].y - v[k].y;
			v2[p++] = Vector2(offset, v[k].y + dx1 * (dy / dx));
		}
		dx1 = dx2;
		dx1in = dx2in;
	}
	m_numVertices = p;
}

void MeshMergeMaterialRepack::ClippedTriangle::computeAreaCentroid() {
	Vector2 *v = m_vertexBuffers[m_activeVertexBuffer];
	v[m_numVertices] = v[0];
	m_area = 0;
	float centroidx = 0, centroidy = 0;
	for (uint32_t k = 0; k < m_numVertices; k++) {
		// http://local.wasp.uwa.edu.au/~pbourke/geometry/polyarea/
		float f = v[k].x * v[k + 1].y - v[k + 1].x * v[k].y;
		m_area += f;
		centroidx += f * (v[k].x + v[k + 1].x);
		centroidy += f * (v[k].y + v[k + 1].y);
	}
	m_area = 0.5f * fabsf(m_area);
	if (m_area == 0) {
		m_centroid = Vector2(0.0f, 0.0f);
	} else {
		m_centroid = Vector2(centroidx / (6 * m_area), centroidy / (6 * m_area));
	}
}

void MeshMergeMaterialRepack::ClippedTriangle::clipAABox(float x0, float y0, float x1, float y1) {
	clipVerticalPlane(x0, -1);
	clipHorizontalPlane(y0, -1);
	clipVerticalPlane(x1, 1);
	clipHorizontalPlane(y1, 1);
	computeAreaCentroid();
}

Vector2 MeshMergeMaterialRepack::ClippedTriangle::centroid() {
	return m_centroid;
}

float MeshMergeMaterialRepack::ClippedTriangle::area() {
	return m_area;
}

MeshMergeMaterialRepack::Triangle::Triangle(const Vector2 &p_v0, const Vector2 &p_v1, const Vector2 &p_v2, const Vector3 &p_t0, const Vector3 &p_t1, const Vector3 &p_t2) {
	// Init vertices.
	this->v1 = p_v0;
	this->v2 = p_v2;
	this->v3 = p_v1;
	// Set barycentric coordinates.
	this->t1 = p_t0;
	this->t2 = p_t2;
	this->t3 = p_t1;
	// make sure every triangle is front facing.
	flipBackface();
	// Compute deltas.
	computeDeltas();
	computeUnitInwardNormals();
}

bool MeshMergeMaterialRepack::Triangle::computeDeltas() {
	Vector2 e0 = v3 - v1;
	Vector2 e1 = v2 - v1;
	Vector3 de0 = t3 - t1;
	Vector3 de1 = t2 - t1;
	float denom = 1.0f / (e0.y * e1.x - e1.y * e0.x);
	if (!std::isfinite(denom)) {
		return false;
	}
	float lambda1 = -e1.y * denom;
	float lambda2 = e0.y * denom;
	float lambda3 = e1.x * denom;
	float lambda4 = -e0.x * denom;
	dx = de0 * lambda1 + de1 * lambda2;
	dy = de0 * lambda3 + de1 * lambda4;
	return true;
}

void MeshMergeMaterialRepack::Triangle::flipBackface() {
	// check if triangle is backfacing, if so, swap two vertices
	if (((v3.x - v1.x) * (v2.y - v1.y) - (v3.y - v1.y) * (v2.x - v1.x)) < 0) {
		Vector2 hv = v1;
		v1 = v2;
		v2 = hv; // swap pos
		Vector3 ht = t1;
		t1 = t2;
		t2 = ht; // swap tex
	}
}

void MeshMergeMaterialRepack::Triangle::computeUnitInwardNormals() {
	n1 = v1 - v2;
	n1 = Vector2(-n1.y, n1.x);
	n1 = n1 * (1.0f / sqrtf(n1.x * n1.x + n1.y * n1.y));
	n2 = v2 - v3;
	n2 = Vector2(-n2.y, n2.x);
	n2 = n2 * (1.0f / sqrtf(n2.x * n2.x + n2.y * n2.y));
	n3 = v3 - v1;
	n3 = Vector2(-n3.y, n3.x);
	n3 = n3 * (1.0f / sqrtf(n3.x * n3.x + n3.y * n3.y));
}

bool MeshMergeMaterialRepack::Triangle::drawAA(SamplingCallback cb, void *param) {
	const float PX_INSIDE = 1.0f / sqrtf(2.0f);
	const float PX_OUTSIDE = -1.0f / sqrtf(2.0f);
	const float BK_SIZE = 8;
	const float BK_INSIDE = sqrtf(BK_SIZE * BK_SIZE / 2.0f);
	const float BK_OUTSIDE = -sqrtf(BK_SIZE * BK_SIZE / 2.0f);

	// Bounding rectangle
	float minx = floorf(MAX(MIN(v1.x, MIN(v2.x, v3.x)), 0.0f));
	float miny = floorf(MAX(MIN(v1.y, MIN(v2.y, v3.y)), 0.0f));
	float maxx = ceilf(MAX(v1.x, MAX(v2.x, v3.x)));
	float maxy = ceilf(MAX(v1.y, MAX(v2.y, v3.y)));

	// Align to texel centers
	minx += 0.5f;
	miny += 0.5f;
	maxx += 0.5f;
	maxy += 0.5f;

	// Half-edge constants
	float C1 = n1.x * (-v1.x) + n1.y * (-v1.y);
	float C2 = n2.x * (-v2.x) + n2.y * (-v2.y);
	float C3 = n3.x * (-v3.x) + n3.y * (-v3.y);

	// Loop through blocks
	for (float y0 = miny; y0 <= maxy; y0 += BK_SIZE) {
		for (float x0 = minx; x0 <= maxx; x0 += BK_SIZE) {
			// Corners of block
			float xc = (x0 + (BK_SIZE - 1) / 2.0f);
			float yc = (y0 + (BK_SIZE - 1) / 2.0f);

			// Evaluate half-space functions
			float aC = C1 + n1.x * xc + n1.y * yc;
			float bC = C2 + n2.x * xc + n2.y * yc;
			float cC = C3 + n3.x * xc + n3.y * yc;

			// Skip block when outside an edge
			if ((aC <= BK_OUTSIDE) || (bC <= BK_OUTSIDE) || (cC <= BK_OUTSIDE)) {
				continue;
			}

			// Calculate initial texture coordinates
			Vector3 texRow = t1 + dy * (y0 - v1.y) + dx * (x0 - v1.x);

			// Accept whole block when totally covered
			if ((aC >= BK_INSIDE) && (bC >= BK_INSIDE) && (cC >= BK_INSIDE)) {
				for (float y = y0; y < y0 + BK_SIZE; y++) {
					Vector3 tex = texRow;
					for (float x = x0; x < x0 + BK_SIZE; x++) {
						if (!cb(param, (int)x, (int)y, tex, dx, dy, 1.0f)) {
							return false;
						}
						tex += dx;
					}
					texRow += dy;
				}
			} else { // Partially covered block
				float CY1 = C1 + n1.x * x0 + n1.y * y0;
				float CY2 = C2 + n2.x * x0 + n2.y * y0;
				float CY3 = C3 + n3.x * x0 + n3.y * y0;

				for (float y = y0; y < y0 + BK_SIZE; y++) {
					float CX1 = CY1;
					float CX2 = CY2;
					float CX3 = CY3;
					Vector3 tex = texRow;

					for (float x = x0; x < x0 + BK_SIZE; x++) {
						Vector3 tex2 = t1 + dx * (x - v1.x) + dy * (y - v1.y);
						if (CX1 >= PX_INSIDE && CX2 >= PX_INSIDE && CX3 >= PX_INSIDE) {
							// pixel completely covered
							if (!cb(param, (int)x, (int)y, tex2, dx, dy, 1.0f)) {
								return false;
							}
						} else if ((CX1 >= PX_OUTSIDE) && (CX2 >= PX_OUTSIDE) && (CX3 >= PX_OUTSIDE)) {
							// triangle partially covers pixel. do clipping.
							ClippedTriangle ct(v1 - Vector2(x, y), v2 - Vector2(x, y), v3 - Vector2(x, y));
							ct.clipAABox(-0.5, -0.5, 0.5, 0.5);
							float area = ct.area();
							if (area > 0.0f) {
								if (!cb(param, (int)x, (int)y, tex2, dx, dy, 0.0f)) {
									return false;
								}
							}
						}
						CX1 += n1.x;
						CX2 += n2.x;
						CX3 += n3.x;
						tex += dx;
					}
					CY1 += n1.y;
					CY2 += n2.y;
					CY3 += n3.y;
					texRow += dy;
				}
			}
		}
	}
	return true;
}
