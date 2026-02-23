/**************************************************************************/
/*  behavior_tree_view.h                                                  */
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

/**
 * behavior_tree_view.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef BEHAVIOR_TREE_VIEW_H
#define BEHAVIOR_TREE_VIEW_H

#include "behavior_tree_data.h"

#ifdef LIMBOAI_MODULE
#include "scene/gui/control.h"
#include "scene/gui/tree.h"
#include "scene/resources/style_box_flat.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/style_box_flat.hpp>
#include <godot_cpp/classes/tree.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class BehaviorTreeView : public Control {
	GDCLASS(BehaviorTreeView, Control);

private:
	Tree *tree;

	struct ThemeCache {
		Ref<StyleBoxFlat> sbf_running;
		Ref<StyleBoxFlat> sbf_success;
		Ref<StyleBoxFlat> sbf_failure;

		Ref<Texture2D> icon_running;
		Ref<Texture2D> icon_success;
		Ref<Texture2D> icon_failure;

		Ref<Font> font_custom_name;

		int tree_inner_margin_top = 0;
		int tree_inner_margin_bottom = 0;
	} theme_cache;

	Vector<uint64_t> collapsed_ids;
	uint64_t last_root_id = 0;

	int last_update_msec = 0;
	int update_interval_msec = 0;
	Ref<BehaviorTreeData> update_data;
	bool update_pending = false;

	void _draw_success_status(Object *p_obj, Rect2 p_rect);
	void _draw_running_status(Object *p_obj, Rect2 p_rect);
	void _draw_failure_status(Object *p_obj, Rect2 p_rect);
	void _draw_fresh(Object *p_obj, Rect2 p_rect) {}
	void _item_collapsed(Object *p_obj);
	void _item_selected();
	double _get_editor_scale() const;

	void _update_tree(const Ref<BehaviorTreeData> &p_data);
	void _add_debug_plan_children(TreeItem *p_item, const Dictionary &p_detail);

protected:
	void _do_update_theme_item_cache();

	void _notification(int p_what);

	static void _bind_methods();

public:
	void clear();
	void update_tree(const Ref<BehaviorTreeData> &p_data);

	void set_update_interval_msec(int p_milliseconds) { update_interval_msec = p_milliseconds; }
	int get_update_interval_msec() const { return update_interval_msec; }

	BehaviorTreeView();
};

#endif // ! BEHAVIOR_TREE_VIEW_H

#endif // ! TOOLS_ENABLED
