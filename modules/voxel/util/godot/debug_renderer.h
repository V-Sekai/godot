/**************************************************************************/
/*  debug_renderer.h                                                      */
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

#include "../containers/std_vector.h"
#include "classes/standard_material_3d.h"
#include "direct_multimesh_instance.h"

namespace zylann::godot {

// Helper to draw 3D primitives every frame for debugging purposes
class DebugRenderer {
public:
	DebugRenderer();
	~DebugRenderer();

	// This class does not uses nodes. Call this first to choose in which world it renders.
	void set_world(World3D *world);

	// Call this before issuing drawing commands
	void begin();

	// Draws a box wireframe.
	// The box's origin is its lower corner. Size is defined by the transform's basis.
	void draw_box(const Transform3D &t, Color8 color);

	// Call this after issuing all drawing commands
	void end();

	void clear();

private:
	// TODO GDX: Can't access RenderingServer in the constructor of a registered class.
	// We have to somehow defer initialization to later. See https://github.com/godotengine/godot-cpp/issues/1179
	void init();
	bool _initialized = false;

	StdVector<DirectMultiMeshInstance::TransformAndColor32> _items;
	Ref<MultiMesh> _multimesh;
	DirectMultiMeshInstance _multimesh_instance;
	// TODO World3D is a reference, do not store it by pointer
	World3D *_world = nullptr;
	bool _inside_block = false;
	PackedFloat32Array _bulk_array;
	Ref<StandardMaterial3D> _material;
};

} // namespace zylann::godot
