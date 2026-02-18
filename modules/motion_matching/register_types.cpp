/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "mm_character.h"

#ifdef TOOLS_ENABLED
#include "editor/animation_post_import_plugin.h"
#include "editor/animation_tree_handler_plugin.h"
#include "editor/mm_data_tab.h"
#include "editor/mm_editor.h"
#include "editor/mm_editor_gizmo_plugin.h"
#include "editor/mm_editor_plugin.h"
#include "editor/mm_visualization_tab.h"
#endif

#include "features/mm_bone_data_feature.h"
#include "features/mm_feature.h"
#include "features/mm_trajectory_feature.h"

#include "modifiers/damped_skeleton_modifier.h"

#include "synchronizers/mm_clamp_synchronizer.h"
#include "synchronizers/mm_mix_synchronizer.h"
#include "synchronizers/mm_rootmotion_synchronizer.h"
#include "synchronizers/mm_synchronizer.h"

#include "mm_animation_library.h"
#include "mm_animation_node.h"
#include "mm_query.h"
#include "mm_trajectory_point.h"

void initialize_motion_matching_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_ABSTRACT_CLASS(MMFeature);
		GDREGISTER_CLASS(MMTrajectoryFeature);
		GDREGISTER_CLASS(MMBoneDataFeature);

		GDREGISTER_CLASS(MMAnimationLibrary);
		GDREGISTER_CLASS(MMAnimationNode);
		GDREGISTER_CLASS(MMQueryInput);

		GDREGISTER_CLASS(MMCharacter);

		GDREGISTER_CLASS(DampedSkeletonModifier);
		GDREGISTER_ABSTRACT_CLASS(MMSynchronizer);
		GDREGISTER_CLASS(MMClampSynchronizer);
		GDREGISTER_CLASS(MMRootMotionSynchronizer);
		GDREGISTER_CLASS(MMMixSynchronizer);
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		GDREGISTER_CLASS(AnimationPostImportPlugin);
		GDREGISTER_CLASS(AnimationTreeHandlerPlugin);
		GDREGISTER_CLASS(MMEditorGizmoPlugin);
		GDREGISTER_CLASS(MMEditor);
		GDREGISTER_CLASS(MMEditorPlugin);
		GDREGISTER_CLASS(MMDataTab);
		GDREGISTER_CLASS(MMVisualizationTab);

		EditorPlugins::add_by_type<MMEditorPlugin>();
		EditorPlugins::add_by_type<AnimationTreeHandlerPlugin>();
	}
#endif
}

void uninitialize_motion_matching_module(ModuleInitializationLevel p_level) {
}
