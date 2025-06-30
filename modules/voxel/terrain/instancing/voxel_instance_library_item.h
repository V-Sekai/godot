/**************************************************************************/
/*  voxel_instance_library_item.h                                         */
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
#include "../../util/containers/std_vector.h"
#include "instance_library_item_listener.h"
#include "voxel_instance_generator.h"

namespace zylann::voxel {

class VoxelInstanceLibraryItem : public Resource {
	GDCLASS(VoxelInstanceLibraryItem, Resource)
public:
	void set_item_name(String p_name);
	String get_item_name() const;

	void set_lod_index(int lod);
	int get_lod_index() const;

	void set_generator(Ref<VoxelInstanceGenerator> generator);
	Ref<VoxelInstanceGenerator> get_generator() const;

	void set_persistent(bool persistent);
	bool is_persistent() const;

	float get_floating_sdf_threshold() const;
	void set_floating_sdf_threshold(const float new_threshold);

	float get_floating_sdf_offset_along_normal() const;
	void set_floating_sdf_offset_along_normal(const float new_offset);

	// Internal

	void add_listener(IInstanceLibraryItemListener *listener, int id);
	void remove_listener(IInstanceLibraryItemListener *listener, int id);

#ifdef TOOLS_ENABLED
	virtual void get_configuration_warnings(PackedStringArray &warnings) const;
#endif

protected:
	void notify_listeners(IInstanceLibraryItemListener::ChangeType change);

private:
	void _on_generator_changed();

	static void _bind_methods();

	// For the user, not used by the engine
	String _name;

	// If a layer is persistent, any change to its instances will be saved if the volume has a stream
	// supporting instances. It will also not generate on top of modified surfaces.
	// If a layer is not persistent, changes won't get saved, and it will keep generating on all compliant
	// surfaces.
	bool _persistent = false;

	// Which LOD of the octree this model will spawn into.
	// Higher means larger distances, but lower precision and density
	int _lod_index = 0;

	Ref<VoxelInstanceGenerator> _generator;

	struct ListenerSlot {
		IInstanceLibraryItemListener *listener;
		int id;

		inline bool operator==(const ListenerSlot &other) const {
			return listener == other.listener && id == other.id;
		}
	};

	StdVector<ListenerSlot> _listeners;
	float _floating_sdf_threshold = 0.0f;
	float _floating_sdf_offset_along_normal = -0.1f;
};

} // namespace zylann::voxel
