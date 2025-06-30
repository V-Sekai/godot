/**************************************************************************/
/*  voxel_range_analysis_dialog.h                                         */
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

#include "../../util/godot/classes/accept_dialog.h"
#include "../../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class CheckBox)
ZN_GODOT_FORWARD_DECLARE(class SpinBox)
ZN_GODOT_FORWARD_DECLARE(class GridContainer)

namespace zylann::voxel {

class VoxelRangeAnalysisDialog : public AcceptDialog {
	GDCLASS(VoxelRangeAnalysisDialog, AcceptDialog)
public:
	VoxelRangeAnalysisDialog();

	AABB get_aabb() const;
	bool is_analysis_enabled() const;

private:
	void _on_enabled_checkbox_toggled(bool enabled);
	void _on_area_spinbox_value_changed(float value);

	void add_row(String text, SpinBox *&sb, GridContainer *container, float defval);

	static void _bind_methods();

	CheckBox *_enabled_checkbox;
	SpinBox *_pos_x_spinbox;
	SpinBox *_pos_y_spinbox;
	SpinBox *_pos_z_spinbox;
	SpinBox *_size_x_spinbox;
	SpinBox *_size_y_spinbox;
	SpinBox *_size_z_spinbox;
};

} // namespace zylann::voxel
