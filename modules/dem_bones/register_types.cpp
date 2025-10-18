/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"
#include "dem_bones_processor.h"

// Temporarily disabled problematic includes
// #include "../../scene/resources/3d/importer_mesh.h"
// #include "../../scene/resources/surface_tool.h"

#ifdef TOOLS_ENABLED
// #include "bake_blend_shapes_plugin.h"
// #include "editor/plugins/editor_plugin.h"
#endif

void initialize_dem_bones_module(ModuleInitializationLevel p_level) {
#ifndef _3D_DISABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Temporarily disable problematic classes to get minimal system working
		// GDREGISTER_CLASS(BlendShapeBake);
		GDREGISTER_CLASS(DemBonesProcessor);
	}
#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		// EditorPlugins::add_by_type<BakeBlendShapesPlugin>();
	}
#endif
#endif
}

void uninitialize_dem_bones_module(ModuleInitializationLevel p_level) {
}
