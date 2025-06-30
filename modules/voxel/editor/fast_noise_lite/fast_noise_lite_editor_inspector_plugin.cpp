/**************************************************************************/
/*  fast_noise_lite_editor_inspector_plugin.cpp                           */
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

#include "fast_noise_lite_editor_inspector_plugin.h"
#include "../../util/noise/fast_noise_lite/fast_noise_lite.h"
#include "../../util/noise/fast_noise_lite/fast_noise_lite_gradient.h"

namespace zylann {

bool ZN_FastNoiseLiteEditorInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	return Object::cast_to<ZN_FastNoiseLite>(p_object) != nullptr ||
			Object::cast_to<ZN_FastNoiseLiteGradient>(p_object) != nullptr;
}

void ZN_FastNoiseLiteEditorInspectorPlugin::_zn_parse_begin(Object *p_object) {
	const ZN_FastNoiseLite *noise_ptr = Object::cast_to<ZN_FastNoiseLite>(p_object);
	if (noise_ptr != nullptr) {
		Ref<ZN_FastNoiseLite> noise(noise_ptr);

		ZN_FastNoiseLiteViewer *viewer = memnew(ZN_FastNoiseLiteViewer);
		viewer->set_noise(noise);
		add_custom_control(viewer);
		return;
	}
	const ZN_FastNoiseLiteGradient *noise_gradient_ptr = Object::cast_to<ZN_FastNoiseLiteGradient>(p_object);
	if (noise_gradient_ptr != nullptr) {
		Ref<ZN_FastNoiseLiteGradient> noise_gradient(noise_gradient_ptr);

		ZN_FastNoiseLiteViewer *viewer = memnew(ZN_FastNoiseLiteViewer);
		viewer->set_noise_gradient(noise_gradient);
		add_custom_control(viewer);
		return;
	}
}

} // namespace zylann
