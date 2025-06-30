/**************************************************************************/
/*  voxel_terrain_editor_task_indicator.h                                 */
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

#include "../../util/containers/fixed_array.h"
#include "../../util/godot/classes/scroll_container.h"
#include "../../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class Label)
ZN_GODOT_FORWARD_DECLARE(class HBoxContainer)

namespace zylann::voxel {

class VoxelTerrainEditorTaskIndicator : public ScrollContainer {
	GDCLASS(VoxelTerrainEditorTaskIndicator, ScrollContainer)
public:
	VoxelTerrainEditorTaskIndicator();

	void _notification(int p_what);
	void update_stats();

private:
	enum StatID {
		STAT_STREAM_TASKS,
		STAT_GENERATE_TASKS,
		STAT_MESH_TASKS,
		STAT_TOTAL_TASKS,
		STAT_MAIN_THREAD_TASKS,
		STAT_TOTAL_MEMORY,
		STAT_VOXEL_MEMORY,
		STAT_COUNT
	};

	// When compiling with GodotCpp, `_bind_methods` is not optional.
	static void _bind_methods() {}

	void create_stat(StatID id, String short_name, String long_name);
	void set_stat(StatID id, int64_t value);
	void set_stat(StatID id, int64_t value, const char *unit);
	void set_stat(StatID id, int64_t value, int64_t value2, const char *unit);

	struct Stat {
		int64_t value = 0;
		int64_t value2 = 0;
		Label *label = nullptr;
	};

	FixedArray<Stat, STAT_COUNT> _stats;
	HBoxContainer *_box_container = nullptr;
};

} // namespace zylann::voxel
