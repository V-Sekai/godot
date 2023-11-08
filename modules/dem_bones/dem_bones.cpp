#include "dem_bones.h"

#include "core/variant/typed_array.h"
#include "dem_bones_extension.h"

#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation_library.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/surface_tool.h"

Error BlendShapeBake::convert_scene(Node *p_scene) {
	List<Node *> queue;
	TypedArray<Node> nodes = p_scene->find_children("*", "AnimationPlayer");
	if (!nodes.size()) {
		return OK;
	}
	AnimationPlayer *ap = cast_to<AnimationPlayer>(nodes[0]);
	if (!ap) {
		return OK;
	}
	List<StringName> animation_names;
	ap->get_animation_list(&animation_names);
	Vector<Ref<Animation>> animations;
	for (const StringName &animation_name : animation_names) {
		Ref<Animation> anim = ap->get_animation(animation_name);
		animations.push_back(anim);
	}
	nodes = p_scene->find_children("*", "MeshInstance3D");
	for (int32_t node_i = 0; node_i < nodes.size(); node_i++) {
		MeshInstance3D *mesh_instance_3d = cast_to<MeshInstance3D>(nodes[node_i]);
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
		String mesh_path = p_scene->get_path_to(mesh_instance_3d);
		Skeleton3D *skeleton = memnew(Skeleton3D);
		if (mesh_instance_3d->get_parent_node_3d()) {
			mesh_instance_3d->get_parent_node_3d()->add_child(skeleton, true);
			skeleton->set_owner(p_scene);
		} else {
			memdelete(skeleton);
			continue;
		}
		if (mesh_instance_3d->get_parent()) {
			mesh_instance_3d->get_parent()->remove_child(mesh_instance_3d);
		}
		skeleton->add_child(mesh_instance_3d, true);
		mesh_instance_3d->set_owner(p_scene);
		mesh_instance_3d->set_skeleton_path(NodePath(".."));
		if (surface_mesh.is_null()) {
			continue;
		}
		if (!surface_mesh->get_blend_shape_count()) {
			continue;
		}
		mesh_instance_3d->set_mesh(Ref<ArrayMesh>());
		Ref<ArrayMesh> mesh;
		mesh.instantiate();
		Ref<SurfaceTool> surface_tool;
		surface_tool.instantiate();
		for (int32_t surface_i = 0; surface_i < surface_mesh->get_surface_count(); surface_i++) {
			surface_tool->clear();
			surface_tool->append_from(surface_mesh, surface_i, Transform3D());
			surface_tool->deindex();
			Array surface_arrays = surface_tool->commit_to_arrays();
			HashMap<NodePath, Vector<Vector3>> blends_arrays;
			NodePath mesh_track;
			Vector<StringName> p_blend_paths;
			Array blend_arrays = surface_mesh->surface_get_blend_shape_arrays(surface_i);
			for (int32_t blend_i = 0; blend_i < surface_mesh->get_blend_shape_count(); blend_i++) {
				String blend_name = surface_mesh->get_blend_shape_name(blend_i);
				Array blend_shape_array = blend_arrays[blend_i];
				blends_arrays[mesh_path + ":" + blend_name] = blend_shape_array[ArrayMesh::ARRAY_VERTEX];
			}
			Dem::DemBonesExt<double, float> bones;
			Dictionary output = bones.convert_blend_shapes_without_bones(skeleton, surface_arrays, surface_arrays[ArrayMesh::ARRAY_VERTEX], blends_arrays, animations);
			animations.clear();
			if (!output.size()) {
				continue;
			}
			Array mesh_array = output["mesh_array"];
			if (!mesh_array.size()) {
				continue;
			}
			Ref<Skin> mesh_skin = output["skin"];
			if (mesh_skin.is_null()) {
				continue;
			}
			surface_tool->clear();
			surface_tool->begin(ArrayMesh::PRIMITIVE_TRIANGLES);
			surface_tool->create_from_triangle_arrays(mesh_array);
			if (output.has("animation_library")) {
				Ref<AnimationLibrary> library = output["animation_library"];
				if (library.is_null()) {
					library.instantiate();
				}
				ap->add_animation_library("Baked Animations " + library->get_name(), library);
			}
			surface_tool->set_skin_weight_count(SurfaceTool::SKIN_8_WEIGHTS);
			mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_tool->commit_to_arrays());
			Skeleton3D *new_skeleton = cast_to<Skeleton3D>(output["skeleton"]);
			skeleton->replace_by(new_skeleton != nullptr ? new_skeleton : memnew(Skeleton3D));
			new_skeleton->set_name(skeleton->get_class());
			mesh_instance_3d->set_skin(mesh_skin);
		}
		mesh_instance_3d->set_mesh(mesh);
	}
	return OK;
}
