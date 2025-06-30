/**************************************************************************/
/*  voxel_engine_gd.h                                                     */
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

#include "voxel_engine.h"

namespace zylann {
class ZN_ThreadedTask;
} // namespace zylann

namespace zylann::voxel::godot {

// Godot-facing singleton class.
// the real class is internal and does not need anything from Object.
class VoxelEngine : public Object {
	GDCLASS(VoxelEngine, Object)
public:
	static VoxelEngine *get_singleton();
	static void create_singleton();
	static void destroy_singleton();

	struct Config {
		zylann::voxel::VoxelEngine::Config inner;
		bool ownership_checks;
	};

	static Config get_config_from_godot();

	VoxelEngine();

	int get_version_major() const;
	int get_version_minor() const;
	int get_version_patch() const;
	Vector3i get_version_v() const;
	String get_version_edition() const;
	String get_version_status() const;
	String get_version_git_hash() const;

	Dictionary get_stats() const;
	void schedule_task(Ref<ZN_ThreadedTask> task);

	int get_thread_count() const;
	void set_thread_count(int count);

#ifdef TOOLS_ENABLED
	void set_editor_camera_info(Vector3 position, Vector3 direction);
	Vector3 get_editor_camera_position() const;
	Vector3 get_editor_camera_direction() const;
#endif

#ifdef VOXEL_TESTS
	void run_tests(Dictionary options_dict);
#endif

private:
	void _on_rendering_server_frame_post_draw();

	bool _b_get_threaded_graphics_resource_building_enabled() const;
	// void _b_set_threaded_graphics_resource_building_enabled(bool enabled);

	static void _bind_methods();

#ifdef TOOLS_ENABLED
	Vector3 _editor_camera_position;
	Vector3 _editor_camera_direction;
#endif
};

} // namespace zylann::voxel::godot
