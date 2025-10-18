/**************************************************************************/
/*  keyframe_reduction_import_plugin.h                                    */
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

#include "editor/import/3d/resource_importer_scene.h"
#include "keyframe_reduce.h"

class KeyframeReductionImportPlugin : public EditorScenePostImportPlugin {
	GDCLASS(KeyframeReductionImportPlugin, EditorScenePostImportPlugin);

public:
	struct ImportOptions {
		bool enable_reduction = true;
		real_t max_error = 0.05f;
		real_t step_size = 0.1f;
		real_t tangent_split_angle_threshold = 45.0f;
		bool use_one_euro_filter = true;
		real_t one_euro_min_cutoff = 1.0f;
		real_t one_euro_beta = 0.1f;
		real_t one_euro_dcutoff = 1.0f;
		real_t conversion_sample_rate = 0.05f; // Sample every 0.05 seconds for track conversion
	};

private:
	ImportOptions options;

	void _convert_track_to_bezier(Ref<Animation> p_animation, int p_track_idx, const ImportOptions &p_options);
	void _reduce_bezier_track(Ref<Animation> p_animation, int p_track_idx, const ImportOptions &p_options);

protected:
	static void _bind_methods();

public:
	virtual void get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) override;
	virtual Variant get_option_visibility(const String &p_path, const String &p_scene_import_type, const String &p_option, const HashMap<StringName, Variant> &p_options) const override;

	virtual void internal_process(EditorScenePostImportPlugin::InternalImportCategory p_category, Node *p_base_scene, Node *p_node, Ref<Resource> p_resource, const Dictionary &p_options) override;

	KeyframeReductionImportPlugin();
};
