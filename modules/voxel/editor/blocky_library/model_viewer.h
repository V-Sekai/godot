/**************************************************************************/
/*  model_viewer.h                                                        */
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

// Required in header for GDExtension builds, due to how virtual methods are implemented
#include "../../util/godot/classes/input_event.h"

#include "../../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class Camera3D);
ZN_GODOT_FORWARD_DECLARE(class SubViewport);

namespace zylann {

class ZN_Axes3DControl;

// Basic SubViewport embedded in a Control for viewing 3D stuff.
// Implements camera controls orbitting around the origin.
// Godot has `MeshEditor` but it is specialized for Mesh resources without access to the hierarchy.
class ZN_ModelViewer : public Control {
	GDCLASS(ZN_ModelViewer, Control)
public:
	ZN_ModelViewer();

	void set_camera_distance(float d);

	// Stuff to view can be instanced under this node
	Node *get_viewer_root_node() const;

#if defined(ZN_GODOT)
	void gui_input(const Ref<InputEvent> &p_event) override;
#elif defined(ZN_GODOT_EXTENSION)
	void _gui_input(const Ref<InputEvent> &p_event) override;
#endif

private:
	void update_camera();

	// When compiling with GodotCpp, `_bind_methods` isn't optional.
	static void _bind_methods() {}

	Camera3D *_camera = nullptr;
	float _pitch = 0.f;
	float _yaw = 0.f;
	float _distance = 1.9f;
	ZN_Axes3DControl *_axes_3d_control = nullptr;
	SubViewport *_viewport;
};

} // namespace zylann
