/**************************************************************************/
/*  voxel_instance_library.h                                              */
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

#include "../../constants/voxel_constants.h"
#include "../../util/containers/fixed_array.h"
#include "../../util/containers/std_map.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/resource.h"
#include "../../util/godot/core/string.h"
#include "../../util/godot/core/string_name.h"
#include "../../util/thread/mutex.h"
#include "instance_library_item_listener.h"

namespace zylann::voxel {

class VoxelInstanceLibraryItem;
class VoxelInstanceGenerator;

// Contains a list of items that can be used by VoxelInstancer, associated with a unique ID
class VoxelInstanceLibrary : public Resource, public IInstanceLibraryItemListener {
	GDCLASS(VoxelInstanceLibrary, Resource)

public:
	static const int MAX_ID = 0xffff;

	~VoxelInstanceLibrary();

	int get_next_available_id();
	void add_item(int p_id, Ref<VoxelInstanceLibraryItem> item);
	void remove_item(int p_id);
	void clear();
	int find_item_by_name(String p_name) const;
	int get_item_count() const;

	// Internal

	const VoxelInstanceLibraryItem *get_item_const(int id) const;
	VoxelInstanceLibraryItem *get_item(int id);

#ifdef TOOLS_ENABLED
	int get_item_id(const VoxelInstanceLibraryItem *item) const;
#endif

	// f(int item_id, VoxelInstanceLibraryItem &item)
	template <typename F>
	void for_each_item(F f) {
		for (auto it = _items.begin(); it != _items.end(); ++it) {
			ZN_ASSERT(it->second.is_valid());
			f(it->first, **it->second);
		}
	}

	template <typename F>
	void for_each_item(F f) const {
		for (auto it = _items.begin(); it != _items.end(); ++it) {
			ZN_ASSERT(it->second.is_valid());
			f(it->first, **it->second);
		}
	}

	template <typename TPredicate>
	int find_item(TPredicate pred) const {
		for (auto it = _items.begin(); it != _items.end(); ++it) {
			ZN_ASSERT(it->second.is_valid());
			if (pred(**it->second)) {
				return it->first;
			}
		}
		return -1;
	}

	void add_listener(IInstanceLibraryItemListener *listener);
	void remove_listener(IInstanceLibraryItemListener *listener);

#ifdef TOOLS_ENABLED
	void get_configuration_warnings(PackedStringArray &warnings) const;

	void set_selected_item_id(int id);
	int get_selected_item_id() const;
#endif

	struct PackedItem {
		Ref<VoxelInstanceGenerator> generator;
		unsigned int id;
	};

	void get_packed_items_at_lod(StdVector<PackedItem> &out_items, unsigned int lod_index) const;

protected:
	void set_item(int id, Ref<VoxelInstanceLibraryItem> item);
	void update_packed_items();

	Ref<VoxelInstanceLibraryItem> _b_get_item(int id) const;
	PackedInt32Array _b_get_all_item_ids() const;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	Array _b_get_data() const;
	void _b_set_data(Array data);

#ifdef TOOLS_ENABLED
	Ref<VoxelInstanceLibraryItem> _b_get_selected_item() const;
	void _b_set_selected_item(Ref<VoxelInstanceLibraryItem> item);
#endif

private:
	void on_library_item_changed(int id, IInstanceLibraryItemListener::ChangeType change) override;
	void notify_listeners(int item_id, IInstanceLibraryItemListener::ChangeType change);

	static void _bind_methods();

	// ID => Item
	// Using a map keeps items ordered, so the last item has highest ID
	StdMap<int, Ref<VoxelInstanceLibraryItem>> _items;

	StdVector<IInstanceLibraryItemListener *> _listeners;

	struct PackedItems {
		struct Lod {
			StdVector<PackedItem> items;
		};
		FixedArray<Lod, constants::MAX_LOD> lods;
		mutable Mutex mutex;
		std::atomic_bool needs_update = false;
	};

	// Packed representation of items for use in procedural generation tasks
	PackedItems _packed_items;

#ifdef TOOLS_ENABLED
	int _selected_item_id = -1;
#endif
};

} // namespace zylann::voxel
