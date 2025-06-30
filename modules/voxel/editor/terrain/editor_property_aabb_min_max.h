/**************************************************************************/
/*  editor_property_aabb_min_max.h                                        */
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
#include "../../util/godot/classes/editor_property.h"
#include "../../util/godot/classes/editor_spin_slider.h"
#include "../../util/macros.h"

namespace zylann {

// Alternative to the default AABB editor which presents it as a minimum and maximum point
class ZN_EditorPropertyAABBMinMax : public zylann::godot::ZN_EditorProperty {
	GDCLASS(ZN_EditorPropertyAABBMinMax, zylann::godot::ZN_EditorProperty);

public:
	ZN_EditorPropertyAABBMinMax();

	void setup(double p_min, double p_max, double p_step, bool p_no_slider, const String &p_suffix = String());

protected:
	void _zn_set_read_only(bool p_read_only) override;
	void _zn_update_property() override;

private:
	void _on_value_changed(double p_val);
	void _notification(int p_what);

	static void _bind_methods();

	FixedArray<EditorSpinSlider *, 6> _spinboxes;
	bool _ignore_value_change = false;
};

} // namespace zylann
