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

#ifndef EDITOR_SCENE_IMPORTER_ALEMBIC_H
#define EDITOR_SCENE_IMPORTER_ALEMBIC_H

#include "core/core_bind.h"
#include "core/io/resource_importer.h"
#include "core/templates/vector.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include <Alembic/AbcCollection/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcGeom/IPolyMesh.h>
#include <Alembic/AbcMaterial/All.h>
#include <Alembic/Util/Export.h>

#include <algorithm>
#include <set>
#include <string>

#include <Alembic/Abc/ErrorHandler.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcGeom/IPolyMesh.h>

#include <iterator>

using namespace Alembic::AbcGeom;

typedef std::set<Abc::chrono_t> SampleTimeSet;
typedef std::map<Abc::chrono_t, Abc::M44d> MatrixSampleMap;

class EditorSceneImporterAlembic : public EditorSceneFormatImporter {
private:
	GDCLASS(EditorSceneImporterAlembic, EditorSceneFormatImporter);
	Node3D *_generate_scene(const String &p_path, const Node *scene, const uint32_t p_flags, int p_bake_fps);
	void _generate_node(const String &p_path, const Node *p_scene, const Node *p_node, Node *p_parent, Node *p_owner, HashSet<String> &r_bone_name, HashSet<String> p_light_names, HashSet<String> p_camera_names, HashMap<Skeleton3D *, MeshInstance3D *> &r_skeletons, const HashMap<String, Transform3D> &p_bone_rests, Vector<MeshInstance3D *> &r_mesh_instances, int32_t &r_mesh_count, Skeleton3D *p_skeleton, const int32_t p_max_bone_weights, HashSet<String> &r_removed_bones);
	void _add_mesh_to_mesh_instance(const Node *p_node, const Node *p_scene, Skeleton3D *s, const String &p_path, MeshInstance3D *p_mesh_instance, Node *p_owner, HashSet<String> &r_bone_name, int32_t &r_mesh_count, int32_t p_max_bone_weights);
	Ref<Texture> _load_texture(const Node *p_scene, String p_path);
	void _calc_tangent_from_mesh(const Ref<ArrayMesh> ai_mesh, int i, int tri_index, int index, PoolColorArray::Write &w);
	void _set_texture_mapping_mode(aiTextureMapMode *map_mode, Ref<Texture> texture);
	void _find_texture_path(const String &p_path, String &path, bool &r_found);
	void _find_texture_path(const String &p_path, _Directory &dir, String &path, bool &found, String extension);
	void _insert_pivot_anim_track(const Vector<MeshInstance3D *> p_meshes, const String p_node_name, Vector<const aiNodeAnim *> F, AnimationPlayer *ap, Skeleton3D *sk, float &length, float ticks_per_second, Ref<Animation> animation, int p_bake_fps, const String &p_path, const Node *p_scene);
	void _register_project_setting_import(const String generic, const String import_setting_string, const Vector<String> &exts, List<String> *r_extensions, const bool p_enabled) const;

protected:
	static void _bind_methods();

public:
	EditorSceneImporterAlembic() {
	}
	~EditorSceneImporterAlembic() {
	}

	virtual void get_extensions(List<String> *r_extensions) const override {
		r_extensions->push_back("abc");
	}
	virtual uint32_t get_import_flags() const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, uint32_t p_compress_flags, List<String> *r_missing_deps, Error *r_err = NULL);

private:
	Ref<Animation> anim = memnew(Animation);
	bool is_leaf(AbcG::IObject iObj);
	bool is_leaf(Abc::ICompoundProperty iProp, Abc::PropertyHeader iHeader);
	int index(Abc::ICompoundProperty iProp, Abc::PropertyHeader iHeader);
	void tree(Abc::IScalarProperty iProp, Node *p_root, std::string prefix = "");
	void tree(Abc::IArrayProperty iProp, Node *p_root, std::string prefix = "");
	void tree(Abc::ICompoundProperty iProp, Node *p_root, std::string prefix = "");
	void tree(AbcG::IObject iObj, Node *p_root, Node *current, bool showProps = false, std::string prefix = "", real_t p_back_fps = 30.0f);
	void GetRelevantSampleTimes(double frame,
			double fps,
			double shutterOpen,
			double shutterClose,
			AbcA::TimeSamplingPtr timeSampling,
			size_t numSamples, SampleTimeSet &output);
	bool is_leaf(AbcG::IObject iObj);
	bool is_leaf(Abc::ICompoundProperty iProp, Abc::PropertyHeader iHeader);
	int index(Abc::ICompoundProperty iProp, Abc::PropertyHeader iHeader);
	void tree(Abc::IScalarProperty iProp, Node *p_root, std::string prefix);
	void GetRelevantSampleTimes(double frame, double fps, double shutterOpen, double shutterClose, AbcA::TimeSamplingPtr timeSampling, size_t numSamples, SampleTimeSet &output);
	void tree(Abc::IArrayProperty iProp, Node *p_root, std::string prefix);
	void tree(Abc::ICompoundProperty iProp, Node *p_root, std::string prefix);
	void tree(AbcG::IObject iObj, Node *p_root, Node *current, std::string prefix, real_t p_back_fps);
};