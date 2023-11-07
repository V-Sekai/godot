#include "dem_bones.h"

#include "dem_bones_extension.h"

#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/animation_library.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/surface_tool.h"

Error BlendShapeBake::convert_scene(Node *p_scene) {
	List<Node *> queue;
	AnimationPlayer *ap = nullptr;
	queue.push_back(p_scene);
	while (!queue.is_empty()) {
		List<Node *>::Element *front = queue.front();
		Node *node = front->get();
		ap = cast_to<AnimationPlayer>(node);
		if (ap) {
			queue.clear();
			break;
		}
		int child_count = node->get_child_count();
		for (int32_t i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
	}
	if (!ap) {
		return OK;
	}
	queue.push_back(p_scene);

	List<StringName> animation_names;
	ap->get_animation_list(&animation_names);
	Vector<Ref<Animation>> animations;
	for (const StringName &animation_name : animation_names) {
		Ref<Animation> anim = ap->get_animation(animation_name);
		animations.push_back(anim);
	}
	while (!queue.is_empty()) {
		List<Node *>::Element *front = queue.front();
		Node *node = front->get();
		int child_count = node->get_child_count();
		for (int32_t i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
		MeshInstance3D *mesh_instance_3d = cast_to<MeshInstance3D>(node);
		if (!mesh_instance_3d) {
			continue;
		}
		// TODO 2021-04-20
		// - To hard-lock the transformations of bones: in the input mesh,
		// create bool attributes for joint nodes (bones) with name "demLock" and
		// set the value to "true".

		// TODO 2021-04-20
		// - To soft-lock skinning weights of vertices: in the input mesh,
		// paint per-vertex colors in gray-scale. The closer the color to white,
		// the more skinning weights of the vertex are preserved.
		Ref<ArrayMesh> surface_mesh = mesh_instance_3d->get_mesh();
		if (surface_mesh.is_null()) {
			continue;
		}
		if (!surface_mesh->get_blend_shape_count()) {
			continue;
		}
		mesh_instance_3d->set_mesh(Ref<ArrayMesh>());
		Ref<SurfaceTool> st;
		st.instantiate();
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		for (int32_t surface_i = 0; surface_i < surface_mesh->get_surface_count(); surface_i++) {
			st->clear();
			st->begin(ArrayMesh::PRIMITIVE_TRIANGLES);
			Array surface_arrays = surface_mesh->surface_get_arrays(surface_i);
			HashMap<String, Vector<Vector3>> blends_arrays;
			NodePath mesh_track;
			Vector<StringName> p_blend_paths;
			String mesh_path = p_scene->get_path_to(mesh_instance_3d);
			for (int32_t blend_i = 0; blend_i < surface_mesh->get_blend_shape_count(); blend_i++) {
				String blend_name = surface_mesh->get_blend_shape_name(blend_i);
				blends_arrays[mesh_path + ":blend_shapes/" + blend_name] = surface_mesh->surface_get_blend_shape_arrays(surface_i)[blend_i];
			}
			Dem::DemBonesExt<double, float> bones;
			Dictionary output = bones.convert_blend_shapes_without_bones(surface_arrays, surface_arrays[ArrayMesh::ARRAY_VERTEX], blends_arrays, animations);
			animations.clear();
			Array mesh_array = output["mesh_array"];
			st->create_from_triangle_arrays(mesh_array);
			if (output.has("animation_library")) {
				Ref<AnimationLibrary> library = output["animation_library"];
				if (library.is_null()) {
					library.instantiate();
				}
				ap->add_animation_library(library->get_name(), library);
			}
			mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, st->commit_to_arrays());
		}
		mesh_instance_3d->set_mesh(mesh);
	}
	return OK;
}
