/**************************************************************************/
/*  spot_noise_editor_inspector_plugin.cpp                                */
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

#include "spot_noise_editor_inspector_plugin.h"
#include "../../util/noise/spot_noise_gd.h"
#include "spot_noise_viewer.h"

namespace zylann {

bool ZN_SpotNoiseEditorInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	return Object::cast_to<ZN_SpotNoise>(p_object) != nullptr;
}

void ZN_SpotNoiseEditorInspectorPlugin::_zn_parse_begin(Object *p_object) {
	const ZN_SpotNoise *noise_ptr = Object::cast_to<ZN_SpotNoise>(p_object);
	if (noise_ptr != nullptr) {
		Ref<ZN_SpotNoise> noise(noise_ptr);

		ZN_SpotNoiseViewer *viewer = memnew(ZN_SpotNoiseViewer);
		viewer->set_noise(noise);
		add_custom_control(viewer);
		return;
	}
}

} // namespace zylann
