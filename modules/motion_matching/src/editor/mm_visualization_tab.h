/**************************************************************************/
/*  mm_visualization_tab.h                                                */
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

#include "mm_animation_library.h"

#include "scene/gui/box_container.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/slider.h" // HSlider
#include "scene/gui/tab_bar.h"

class MMVisualizationTab : public TabBar {
	GDCLASS(MMVisualizationTab, TabBar)

public:
	void set_animation_library(Ref<MMAnimationLibrary> p_library) {
		_current_animation_library = p_library;
		refresh();
	}

	void set_enabled(bool p_enabled);
	void refresh();
	void clear();

protected:
	static void _bind_methods();
	void _notification(int p_notification);

private:
	void _viz_time_changed(float p_value);
	void _viz_anim_selected(int p_index);
	void _emit_animation_viz_request(String p_animation_lib, String p_animation_name, int32_t p_pose_index);

	Ref<MMAnimationLibrary> _current_animation_library;

	// Visualization
	Label *_warning_label;
	VBoxContainer *_visualization_vbox;
	OptionButton *_viz_animation_option_button;
	HSlider *_viz_time_slider;
	int _selected_animation_index = -1;
};
