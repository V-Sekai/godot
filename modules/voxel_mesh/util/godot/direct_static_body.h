/**************************************************************************/
/*  direct_static_body.h                                                  */
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

#ifndef DIRECT_STATIC_BODY_H
#define DIRECT_STATIC_BODY_H

#include "direct_mesh_instance.h"

#include <core/templates/rid.h>
#include <scene/resources/shape_3d.h>

class World;

// Thin wrapper around static body API
class DirectStaticBody {
public:
	DirectStaticBody();
	~DirectStaticBody();

	void create();
	void destroy();
	bool is_valid() const;
	void set_transform(Transform3D transform);
	void add_shape(Ref<Shape3D> shape);
	void remove_shape(int shape_index);
	Ref<Shape3D> get_shape(int shape_index);
	void set_world(World3D *world);
	void set_shape_enabled(int shape_index, bool disabled);
	void set_attached_object(Object *obj);
	void set_collision_layer(int layer);
	void set_collision_mask(int mask);

	void set_debug(bool enabled, World3D *world);

private:
	RID _body;
	Ref<Shape3D> _shape;
	DirectMeshInstance _debug_mesh_instance;
};

#endif // DIRECT_STATIC_BODY_H
