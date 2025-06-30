/**************************************************************************/
/*  spot_noise_viewer.h                                                   */
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

#include "../../util/godot/classes/control.h"
#include "../../util/godot/macros.h"
#include "../../util/noise/spot_noise_gd.h"

ZN_GODOT_FORWARD_DECLARE(class TextureRect)

namespace zylann {

class ZN_SpotNoiseViewer : public Control {
	GDCLASS(ZN_SpotNoiseViewer, Control)
public:
	static const int PREVIEW_WIDTH = 300;
	static const int PREVIEW_HEIGHT = 150;

	ZN_SpotNoiseViewer();

	void set_noise(Ref<ZN_SpotNoise> noise);

private:
	void _on_noise_changed();
	void _notification(int p_what);

	void update_preview();

	static void _bind_methods();

	Ref<ZN_SpotNoise> _noise;
	float _time_before_update = -1.f;
	TextureRect *_texture_rect = nullptr;
};

} // namespace zylann
