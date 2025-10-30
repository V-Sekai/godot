/**************************************************************************/
/*  erlang_variant.h                                                      */
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

#ifdef LINUXBSD_ENABLED

#include "core/object/object.h"
#include "core/variant/variant.h"
#include <ei.h>
#include <vector>

// Per-connection scoped object management (following godot-sandbox pattern)
class BeamObjectScope {
private:
	std::vector<Object *> scoped_objects;

public:
	BeamObjectScope() = default;
	~BeamObjectScope() = default;

	// Add object to scope, return its scoped address/handle
	void add_scoped_object(Object *obj);

	// Get object from scoped address
	Object *get_object(uintptr_t addr) const;

	// Check if address points to a scoped object
	bool is_object_scoped(uintptr_t addr) const;

	// Clear all scoped objects
	void clear();

	// Get all scoped objects
	const std::vector<Object *> &get_scoped_objects() const { return scoped_objects; }
};

// Erlang variant encoding/decoding (following godot-sandbox GuestVariant pattern)
class ErlangVariantEncoder {
public:
	// Encode a Godot Variant into ei_x_buff format for Erlang
	static void encode_variant(ei_x_buff *buf, const Variant &var, BeamObjectScope *scope = nullptr);

private:
	static void encode_basic_type(ei_x_buff *buf, const Variant &var);
	static void encode_vector2(ei_x_buff *buf, const Vector2 &v);
	static void encode_vector2i(ei_x_buff *buf, const Vector2i &v);
	static void encode_rect2(ei_x_buff *buf, const Rect2 &r);
	static void encode_rect2i(ei_x_buff *buf, const Rect2i &r);
	static void encode_vector3(ei_x_buff *buf, const Vector3 &v);
	static void encode_vector3i(ei_x_buff *buf, const Vector3i &v);
	static void encode_transform2d(ei_x_buff *buf, const Transform2D &t);
	static void encode_vector4(ei_x_buff *buf, const Vector4 &v);
	static void encode_vector4i(ei_x_buff *buf, const Vector4i &v);
	static void encode_plane(ei_x_buff *buf, const Plane &p);
	static void encode_quaternion(ei_x_buff *buf, const Quaternion &q);
	static void encode_aabb(ei_x_buff *buf, const AABB &a);
	static void encode_basis(ei_x_buff *buf, const Basis &b);
	static void encode_transform3d(ei_x_buff *buf, const Transform3D &t);
	static void encode_projection(ei_x_buff *buf, const Projection &p);
	static void encode_color(ei_x_buff *buf, const Color &c);
	static void encode_string_name(ei_x_buff *buf, const StringName &s);
	static void encode_node_path(ei_x_buff *buf, const NodePath &n);
	static void encode_rid(ei_x_buff *buf, const RID &r);
	static void encode_string(ei_x_buff *buf, const String &s);
	static void encode_callable(ei_x_buff *buf, const Callable &c);
	static void encode_signal(ei_x_buff *buf, const Signal &s);
	static void encode_dictionary(ei_x_buff *buf, const Dictionary &d, BeamObjectScope *scope);
	static void encode_array(ei_x_buff *buf, const Array &a, BeamObjectScope *scope);
	static void encode_packed_array(ei_x_buff *buf, const Variant &var);
	static void encode_object(ei_x_buff *buf, Object *obj, BeamObjectScope *scope);
};

// Erlang variant decoding
class ErlangVariantDecoder {
public:
	// Decode from ei_x_buff to Godot Variant
	static Variant decode_variant(ei_x_buff *buf, int *index, BeamObjectScope *scope = nullptr);

private:
	static Variant decode_tuple(ei_x_buff *buf, int *index, int arity, BeamObjectScope *scope);
	static Variant decode_vector2(ei_x_buff *buf, int *index);
	static Variant decode_vector2i(ei_x_buff *buf, int *index);
	static Variant decode_rect2(ei_x_buff *buf, int *index);
	static Variant decode_rect2i(ei_x_buff *buf, int *index);
	static Variant decode_vector3(ei_x_buff *buf, int *index);
	static Variant decode_vector3i(ei_x_buff *buf, int *index);
	static Variant decode_transform2d(ei_x_buff *buf, int *index);
	static Variant decode_vector4(ei_x_buff *buf, int *index);
	static Variant decode_vector4i(ei_x_buff *buf, int *index);
	static Variant decode_plane(ei_x_buff *buf, int *index);
	static Variant decode_quaternion(ei_x_buff *buf, int *index);
	static Variant decode_aabb(ei_x_buff *buf, int *index);
	static Variant decode_basis(ei_x_buff *buf, int *index);
	static Variant decode_transform3d(ei_x_buff *buf, int *index);
	static Variant decode_projection(ei_x_buff *buf, int *index);
	static Variant decode_color(ei_x_buff *buf, int *index);
	static Variant decode_string_name(ei_x_buff *buf, int *index);
	static Variant decode_node_path(ei_x_buff *buf, int *index);
	static Variant decode_rid(ei_x_buff *buf, int *index);
	static Variant decode_callable(ei_x_buff *buf, int *index);
	static Variant decode_signal(ei_x_buff *buf, int *index);
	static Variant decode_dictionary(ei_x_buff *buf, int *index, BeamObjectScope *scope);
	static Variant decode_array(ei_x_buff *buf, int *index, BeamObjectScope *scope);
	static Variant decode_packed_array(ei_x_buff *buf, int *index, const char *type_atom);
	static Variant decode_object(ei_x_buff *buf, int *index, BeamObjectScope *scope);
};

#endif // LINUXBSD_ENABLED
