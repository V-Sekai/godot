#pragma once

#ifdef TOOLS_ENABLED

#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/gui/editor_file_dialog.h"
#include "modules/scene_merge/scene_merge.h"
#include "scene/gui/check_box.h"

#include "scene/resources/packed_scene.h"

#include "merge.h"

class SceneMergePlugin : public EditorPlugin {

	GDCLASS(SceneMergePlugin, EditorPlugin);
	Ref<SceneMerge> scene_optimize;
	void _action();

public:
	SceneMergePlugin();
	~SceneMergePlugin();
};

#endif // MERGE_PLUGIN_H
