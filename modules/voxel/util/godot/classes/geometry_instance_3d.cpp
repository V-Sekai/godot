/**************************************************************************/
/*  geometry_instance_3d.cpp                                              */
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

#include "geometry_instance_3d.h"
#include "rendering_server.h"

namespace zylann::godot {

const char *const CAST_SHADOW_ENUM_HINT_STRING = "Off,On,Double-Sided,Shadows Only";
const char *const GI_MODE_ENUM_HINT_STRING = "Disabled,Static (VoxelGI/SDFGI/LightmapGI),Dynamic (VoxelGI only)";

void set_geometry_instance_gi_mode(RID rid, GeometryInstance3D::GIMode mode) {
	ERR_FAIL_COND(!rid.is_valid());
	RenderingServer &vs = *RenderingServer::get_singleton();

	bool baked_light;
	bool dynamic_gi;

	switch (mode) {
		case GeometryInstance3D::GI_MODE_DISABLED:
			baked_light = false;
			dynamic_gi = false;
			break;
		case GeometryInstance3D::GI_MODE_STATIC:
			baked_light = true;
			dynamic_gi = false;
			break;
		case GeometryInstance3D::GI_MODE_DYNAMIC:
			baked_light = false;
			dynamic_gi = true;
			break;
		default:
			ERR_FAIL_MSG("Unexpected GIMode");
			return;
	}

	vs.instance_geometry_set_flag(rid, RenderingServer::INSTANCE_FLAG_USE_BAKED_LIGHT, baked_light);
	vs.instance_geometry_set_flag(rid, RenderingServer::INSTANCE_FLAG_USE_DYNAMIC_GI, dynamic_gi);
}

} // namespace zylann::godot
