/**************************************************************************/
/*  mm_data_tab.cpp                                                       */
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

#include "mm_data_tab.h"

#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/scroll_container.h"

void MMDataTab::set_animation_library(Ref<MMAnimationLibrary> p_library) {
	if (p_library.is_null()) {
		return;
	}
	if (_stats_data_container == nullptr) {
		return; // UI not built yet (ENTER_TREE not run).
	}

	_clear_data();

	if (p_library->needs_baking()) {
		return;
	}

	Label *min_label = memnew(Label);
	min_label->set_text("Min");
	_stats_data_container->add_child(min_label);

	Label *max_label = memnew(Label);
	max_label->set_text("Max");
	_stats_data_container->add_child(max_label);

	Label *avg_label = memnew(Label);
	avg_label->set_text("Avg");
	_stats_data_container->add_child(avg_label);

	Label *std_label = memnew(Label);
	std_label->set_text("Std");
	_stats_data_container->add_child(std_label);

	TypedArray<MMFeature> features = p_library->get_features();
	for (int i = 0; i < features.size(); i++) {
		Ref<MMFeature> feature = features[i];
		if (feature.is_null()) {
			return;
		}

		for (int j = 0; j < feature->get_dimension_count(); j++) {
			Label *min_value = memnew(Label);
			min_value->set_text(String::num(feature->get_mins()[j]));
			_stats_data_container->add_child(min_value);

			Label *max_value = memnew(Label);
			max_value->set_text(String::num(feature->get_maxes()[j]));
			_stats_data_container->add_child(max_value);

			Label *avg_value = memnew(Label);
			avg_value->set_text(String::num(feature->get_means()[j]));
			_stats_data_container->add_child(avg_value);

			Label *std_value = memnew(Label);
			std_value->set_text(String::num(feature->get_std_devs()[j]));
			_stats_data_container->add_child(std_value);
		}
	}
}

void MMDataTab::clear() {
	_clear_data();
}

void MMDataTab::_bind_methods() {
}

void MMDataTab::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE: {
			// Build UI only once; ENTER_TREE can run again when switching tabs.
			if (_stats_data_container != nullptr) {
				return;
			}
			set_name("Data");

			ScrollContainer *scroll_container = memnew(ScrollContainer);
			scroll_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
			add_child(scroll_container);

			VBoxContainer *main_container = memnew(VBoxContainer);
			scroll_container->add_child(main_container);

			_stats_data_container = memnew(GridContainer);
			_stats_data_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
			_stats_data_container->set_columns(4); // min, max, avg, std
			main_container->add_child(_stats_data_container);

			_motion_data_container = memnew(GridContainer);
			_motion_data_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
			main_container->add_child(_motion_data_container);

		} break;
	}
}

void MMDataTab::_clear_data() {
	if (_stats_data_container == nullptr) {
		return;
	}
	TypedArray<Node> stat_cells = _stats_data_container->get_children();
	for (int i = 0; i < stat_cells.size(); i++) {
		Node *cell = Object::cast_to<Node>(stat_cells[i]);
		cell->queue_free();
	}
}
