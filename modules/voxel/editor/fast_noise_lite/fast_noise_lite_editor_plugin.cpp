/**************************************************************************/
/*  fast_noise_lite_editor_plugin.cpp                                     */
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

#include "fast_noise_lite_editor_plugin.h"
#include "../../util/noise/fast_noise_lite/fast_noise_lite.h"
#include "../../util/noise/fast_noise_lite/fast_noise_lite_gradient.h"
#include "fast_noise_lite_editor_inspector_plugin.h"
#include "fast_noise_lite_viewer.h"

namespace zylann {

ZN_FastNoiseLiteEditorPlugin::ZN_FastNoiseLiteEditorPlugin() {
	Ref<ZN_FastNoiseLiteEditorInspectorPlugin> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}

String ZN_FastNoiseLiteEditorPlugin::_zn_get_plugin_name() const {
	return ZN_FastNoiseLite::get_class_static();
}

} // namespace zylann
