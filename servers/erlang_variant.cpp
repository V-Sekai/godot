/**************************************************************************/
/*  erlang_variant.cpp                                                    */
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

#ifdef LINUXBSD_ENABLED

#include "erlang_variant.h"

#include "core/string/ustring.h"
#include "core/string/node_path.h"
#include "scene/main/node.h"
#include "core/math/aabb.h"
#include "core/templates/rid.h"
#include <iostream>

// BeamObjectScope implementation

void BeamObjectScope::add_scoped_object(Object *obj) {
	if (obj == nullptr) {
		return;
	}
	scoped_objects.push_back(obj);
}

Object *BeamObjectScope::get_object(uintptr_t addr) const {
	// Search for object pointer in scoped list
	for (Object *obj : scoped_objects) {
		if (reinterpret_cast<uintptr_t>(obj) == addr) {
			return obj;
		}
	}
	return nullptr;
}

bool BeamObjectScope::is_object_scoped(uintptr_t addr) const {
	return get_object(addr) != nullptr;
}

void BeamObjectScope::clear() {
	scoped_objects.clear();
}

// ErlangVariantEncoder implementation

void ErlangVariantEncoder::encode_variant(ei_x_buff *buf, const Variant &var, BeamObjectScope *scope) {
	switch (var.get_type()) {
		case Variant::NIL:
		case Variant::BOOL:
		case Variant::INT:
		case Variant::FLOAT:
			encode_basic_type(buf, var);
			break;
		case Variant::VECTOR2:
			encode_vector2(buf, var.operator Vector2());
			break;
		case Variant::VECTOR2I:
			encode_vector2i(buf, var.operator Vector2i());
			break;
		case Variant::RECT2:
			encode_rect2(buf, var.operator Rect2());
			break;
		case Variant::RECT2I:
			encode_rect2i(buf, var.operator Rect2i());
			break;
		case Variant::VECTOR3:
			encode_vector3(buf, var.operator Vector3());
			break;
		case Variant::VECTOR3I:
			encode_vector3i(buf, var.operator Vector3i());
			break;
		case Variant::TRANSFORM2D:
			encode_transform2d(buf, var.operator Transform2D());
			break;
		case Variant::VECTOR4:
			encode_vector4(buf, var.operator Vector4());
			break;
		case Variant::VECTOR4I:
			encode_vector4i(buf, var.operator Vector4i());
			break;
		case Variant::PLANE:
			encode_plane(buf, var.operator Plane());
			break;
		case Variant::QUATERNION:
			encode_quaternion(buf, var.operator Quaternion());
			break;
		case Variant::AABB:
			encode_aabb(buf, var.operator ::AABB());
			break;
		case Variant::BASIS:
			encode_basis(buf, var.operator Basis());
			break;
		case Variant::TRANSFORM3D:
			encode_transform3d(buf, var.operator Transform3D());
			break;
		case Variant::PROJECTION:
			encode_projection(buf, var.operator Projection());
			break;
		case Variant::COLOR:
			encode_color(buf, var.operator Color());
			break;
		case Variant::STRING_NAME:
			encode_string_name(buf, var.operator StringName());
			break;
		case Variant::NODE_PATH:
			encode_node_path(buf, var.operator NodePath());
			break;
		case Variant::RID:
			encode_rid(buf, var.operator ::RID());
			break;
		case Variant::STRING:
			encode_string(buf, var.operator String());
			break;
		case Variant::CALLABLE:
			encode_callable(buf, var.operator Callable());
			break;
		case Variant::SIGNAL:
			encode_signal(buf, var.operator Signal());
			break;
		case Variant::DICTIONARY:
			encode_dictionary(buf, var.operator Dictionary(), scope);
			break;
		case Variant::ARRAY:
			encode_array(buf, var.operator Array(), scope);
			break;
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
		case Variant::PACKED_STRING_ARRAY:
		case Variant::PACKED_VECTOR2_ARRAY:
		case Variant::PACKED_VECTOR3_ARRAY:
		case Variant::PACKED_COLOR_ARRAY:
		case Variant::PACKED_VECTOR4_ARRAY:
			encode_packed_array(buf, var);
			break;
		case Variant::OBJECT:
			encode_object(buf, var.operator Object *(), scope);
			break;
		default:
			encode_basic_type(buf, var); // fallback for unsupported types
			break;
	}
}

void ErlangVariantEncoder::encode_basic_type(ei_x_buff *buf, const Variant &var) {
	switch (var.get_type()) {
		case Variant::NIL:
			ei_x_encode_atom(buf, "nil");
			break;
		case Variant::BOOL: {
			bool b = var.operator bool();
			ei_x_encode_atom(buf, b ? "true" : "false");
			break;
		}
		case Variant::INT: {
			int64_t i = var.operator int64_t();
			ei_x_encode_long(buf, i);
			break;
		}
		case Variant::FLOAT: {
			double f = var.operator double();
			ei_x_encode_double(buf, f);
			break;
		}
		default:
			ei_x_encode_atom(buf, "unsupported_type");
			break;
	}
}

void ErlangVariantEncoder::encode_vector2(ei_x_buff *buf, const Vector2 &v) {
	ei_x_encode_tuple_header(buf, 3);
	ei_x_encode_atom(buf, "vector2");
	ei_x_encode_double(buf, v.x);
	ei_x_encode_double(buf, v.y);
}

void ErlangVariantEncoder::encode_vector3(ei_x_buff *buf, const Vector3 &v) {
	ei_x_encode_tuple_header(buf, 4);
	ei_x_encode_atom(buf, "vector3");
	ei_x_encode_double(buf, v.x);
	ei_x_encode_double(buf, v.y);
	ei_x_encode_double(buf, v.z);
}

void ErlangVariantEncoder::encode_vector4(ei_x_buff *buf, const Vector4 &v) {
	ei_x_encode_tuple_header(buf, 5);
	ei_x_encode_atom(buf, "vector4");
	ei_x_encode_double(buf, v.x);
	ei_x_encode_double(buf, v.y);
	ei_x_encode_double(buf, v.z);
	ei_x_encode_double(buf, v.w);
}

void ErlangVariantEncoder::encode_color(ei_x_buff *buf, const Color &c) {
	ei_x_encode_tuple_header(buf, 5);
	ei_x_encode_atom(buf, "color");
	ei_x_encode_double(buf, c.r);
	ei_x_encode_double(buf, c.g);
	ei_x_encode_double(buf, c.b);
	ei_x_encode_double(buf, c.a);
}

void ErlangVariantEncoder::encode_string(ei_x_buff *buf, const String &s) {
	CharString utf8 = s.utf8();
	ei_x_encode_binary(buf, utf8.ptr(), utf8.length());
}

void ErlangVariantEncoder::encode_vector2i(ei_x_buff *buf, const Vector2i &v) {
	ei_x_encode_tuple_header(buf, 3);
	ei_x_encode_atom(buf, "vector2i");
	ei_x_encode_long(buf, v.x);
	ei_x_encode_long(buf, v.y);
}

void ErlangVariantEncoder::encode_rect2(ei_x_buff *buf, const Rect2 &r) {
	ei_x_encode_tuple_header(buf, 5);
	ei_x_encode_atom(buf, "rect2");
	ei_x_encode_double(buf, r.position.x);
	ei_x_encode_double(buf, r.position.y);
	ei_x_encode_double(buf, r.size.x);
	ei_x_encode_double(buf, r.size.y);
}

void ErlangVariantEncoder::encode_rect2i(ei_x_buff *buf, const Rect2i &r) {
	ei_x_encode_tuple_header(buf, 5);
	ei_x_encode_atom(buf, "rect2i");
	ei_x_encode_long(buf, r.position.x);
	ei_x_encode_long(buf, r.position.y);
	ei_x_encode_long(buf, r.size.x);
	ei_x_encode_long(buf, r.size.y);
}

void ErlangVariantEncoder::encode_vector3i(ei_x_buff *buf, const Vector3i &v) {
	ei_x_encode_tuple_header(buf, 4);
	ei_x_encode_atom(buf, "vector3i");
	ei_x_encode_long(buf, v.x);
	ei_x_encode_long(buf, v.y);
	ei_x_encode_long(buf, v.z);
}

void ErlangVariantEncoder::encode_transform2d(ei_x_buff *buf, const Transform2D &t) {
	ei_x_encode_tuple_header(buf, 7);
	ei_x_encode_atom(buf, "transform2d");
	ei_x_encode_double(buf, t.columns[0].x);
	ei_x_encode_double(buf, t.columns[0].y);
	ei_x_encode_double(buf, t.columns[1].x);
	ei_x_encode_double(buf, t.columns[1].y);
	Vector2 origin = t.get_origin();
	ei_x_encode_double(buf, origin.x);
	ei_x_encode_double(buf, origin.y);
}

void ErlangVariantEncoder::encode_vector4i(ei_x_buff *buf, const Vector4i &v) {
	ei_x_encode_tuple_header(buf, 5);
	ei_x_encode_atom(buf, "vector4i");
	ei_x_encode_long(buf, v.x);
	ei_x_encode_long(buf, v.y);
	ei_x_encode_long(buf, v.z);
	ei_x_encode_long(buf, v.w);
}

void ErlangVariantEncoder::encode_plane(ei_x_buff *buf, const Plane &p) {
	ei_x_encode_tuple_header(buf, 5);
	ei_x_encode_atom(buf, "plane");
	ei_x_encode_double(buf, p.normal.x);
	ei_x_encode_double(buf, p.normal.y);
	ei_x_encode_double(buf, p.normal.z);
	ei_x_encode_double(buf, p.d);
}

void ErlangVariantEncoder::encode_quaternion(ei_x_buff *buf, const Quaternion &q) {
	ei_x_encode_tuple_header(buf, 5);
	ei_x_encode_atom(buf, "quaternion");
	ei_x_encode_double(buf, q.x);
	ei_x_encode_double(buf, q.y);
	ei_x_encode_double(buf, q.z);
	ei_x_encode_double(buf, q.w);
}

void ErlangVariantEncoder::encode_aabb(ei_x_buff *buf, const AABB &a) {
	ei_x_encode_tuple_header(buf, 7);
	ei_x_encode_atom(buf, "aabb");
	ei_x_encode_double(buf, a.position.x);
	ei_x_encode_double(buf, a.position.y);
	ei_x_encode_double(buf, a.position.z);
	ei_x_encode_double(buf, a.size.x);
	ei_x_encode_double(buf, a.size.y);
	ei_x_encode_double(buf, a.size.z);
}

void ErlangVariantEncoder::encode_basis(ei_x_buff *buf, const Basis &b) {
	ei_x_encode_tuple_header(buf, 10);
	ei_x_encode_atom(buf, "basis");
	ei_x_encode_double(buf, b.rows[0].x);
	ei_x_encode_double(buf, b.rows[0].y);
	ei_x_encode_double(buf, b.rows[0].z);
	ei_x_encode_double(buf, b.rows[1].x);
	ei_x_encode_double(buf, b.rows[1].y);
	ei_x_encode_double(buf, b.rows[1].z);
	ei_x_encode_double(buf, b.rows[2].x);
	ei_x_encode_double(buf, b.rows[2].y);
	ei_x_encode_double(buf, b.rows[2].z);
}

void ErlangVariantEncoder::encode_transform3d(ei_x_buff *buf, const Transform3D &t) {
	ei_x_encode_tuple_header(buf, 13);
	ei_x_encode_atom(buf, "transform3d");
	ei_x_encode_double(buf, t.basis.rows[0].x);
	ei_x_encode_double(buf, t.basis.rows[0].y);
	ei_x_encode_double(buf, t.basis.rows[0].z);
	ei_x_encode_double(buf, t.basis.rows[1].x);
	ei_x_encode_double(buf, t.basis.rows[1].y);
	ei_x_encode_double(buf, t.basis.rows[1].z);
	ei_x_encode_double(buf, t.basis.rows[2].x);
	ei_x_encode_double(buf, t.basis.rows[2].y);
	ei_x_encode_double(buf, t.basis.rows[2].z);
	ei_x_encode_double(buf, t.origin.x);
	ei_x_encode_double(buf, t.origin.y);
	ei_x_encode_double(buf, t.origin.z);
}

void ErlangVariantEncoder::encode_projection(ei_x_buff *buf, const Projection &p) {
	ei_x_encode_tuple_header(buf, 17);
	ei_x_encode_atom(buf, "projection");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			ei_x_encode_double(buf, p.columns[i][j]);
		}
	}
}

void ErlangVariantEncoder::encode_string_name(ei_x_buff *buf, const StringName &s) {
	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, "string_name");
	String str = s;
	CharString utf8 = str.utf8();
	ei_x_encode_binary(buf, utf8.ptr(), utf8.length());
}

void ErlangVariantEncoder::encode_node_path(ei_x_buff *buf, const NodePath &n) {
	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, "node_path");
	String str = n.operator String();
	CharString utf8 = str.utf8();
	ei_x_encode_binary(buf, utf8.ptr(), utf8.length());
}

void ErlangVariantEncoder::encode_rid(ei_x_buff *buf, const RID &r) {
	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, "rid");
	ei_x_encode_long(buf, r.get_id());
}

void ErlangVariantEncoder::encode_callable(ei_x_buff *buf, const Callable &c) {
	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, "callable");
	// For now, just encode as unsupported - full callable support would be complex
	ei_x_encode_atom(buf, "unsupported");
}

void ErlangVariantEncoder::encode_signal(ei_x_buff *buf, const Signal &s) {
	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, "signal");
	// For now, just encode as unsupported - full signal support would be complex
	ei_x_encode_atom(buf, "unsupported");
}

void ErlangVariantEncoder::encode_dictionary(ei_x_buff *buf, const Dictionary &d, BeamObjectScope *scope) {
	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, "dictionary");
	ei_x_encode_list_header(buf, d.size());

	const Variant *key = nullptr;
	while ((key = d.next(key))) {
		ei_x_encode_tuple_header(buf, 2);
		encode_variant(buf, *key, scope);
		encode_variant(buf, d[*key], scope);
	}
	ei_x_encode_empty_list(buf);
}

void ErlangVariantEncoder::encode_array(ei_x_buff *buf, const Array &a, BeamObjectScope *scope) {
	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, "array");
	ei_x_encode_list_header(buf, a.size());

	for (int i = 0; i < a.size(); i++) {
		encode_variant(buf, a[i], scope);
	}
	ei_x_encode_empty_list(buf);
}

void ErlangVariantEncoder::encode_packed_array(ei_x_buff *buf, const Variant &var) {
	const char *type_name = nullptr;
	switch (var.get_type()) {
		case Variant::PACKED_BYTE_ARRAY: type_name = "packed_byte_array"; break;
		case Variant::PACKED_INT32_ARRAY: type_name = "packed_int32_array"; break;
		case Variant::PACKED_INT64_ARRAY: type_name = "packed_int64_array"; break;
		case Variant::PACKED_FLOAT32_ARRAY: type_name = "packed_float32_array"; break;
		case Variant::PACKED_FLOAT64_ARRAY: type_name = "packed_float64_array"; break;
		case Variant::PACKED_STRING_ARRAY: type_name = "packed_string_array"; break;
		case Variant::PACKED_VECTOR2_ARRAY: type_name = "packed_vector2_array"; break;
		case Variant::PACKED_VECTOR3_ARRAY: type_name = "packed_vector3_array"; break;
		case Variant::PACKED_COLOR_ARRAY: type_name = "packed_color_array"; break;
		case Variant::PACKED_VECTOR4_ARRAY: type_name = "packed_vector4_array"; break;
		default: type_name = "unknown_packed_array"; break;
	}

	ei_x_encode_tuple_header(buf, 2);
	ei_x_encode_atom(buf, type_name);
	// For now, encode as empty list - full packed array support would require more complex encoding
	ei_x_encode_empty_list(buf);
}

void ErlangVariantEncoder::encode_object(ei_x_buff *buf, Object *obj, BeamObjectScope *scope) {
	if (obj == nullptr) {
		ei_x_encode_atom(buf, "nil");
		return;
	}

	if (scope) {
		scope->add_scoped_object(obj);
	}

	ei_x_encode_tuple_header(buf, 3);
	ei_x_encode_atom(buf, "object");
	ei_x_encode_long(buf, reinterpret_cast<int64_t>(obj));

	String class_name = obj->get_class();
	CharString class_utf8 = class_name.utf8();
	ei_x_encode_binary(buf, class_utf8.ptr(), class_utf8.length());
}

// ErlangVariantDecoder implementation

Variant ErlangVariantDecoder::decode_variant(ei_x_buff *buf, int *index, BeamObjectScope *scope) {
	int arity;
	char atom[256];

	// Try to decode as atom
	if (ei_decode_atom(buf->buff, index, atom) == 0) {
		if (strcmp(atom, "nil") == 0) {
			return Variant();
		} else if (strcmp(atom, "true") == 0) {
			return Variant(true);
		} else if (strcmp(atom, "false") == 0) {
			return Variant(false);
		}
		// If it's another atom, return it as a string
		return Variant(String::utf8(atom));
	}

	// Reset index and try integer
	long long_arity = 0;
	if (ei_decode_long(buf->buff, index, &long_arity) == 0) {
		return Variant((int64_t)long_arity);
	}

	// Reset index and try double
	double double_val = 0.0;
	if (ei_decode_double(buf->buff, index, &double_val) == 0) {
		return Variant(double_val);
	}

	// Reset index and try tuple
	if (ei_decode_tuple_header(buf->buff, index, &arity) == 0) {
		return decode_tuple(buf, index, arity, scope);
	}

	// Reset index and try binary/string
	long len = 0;
	if (ei_decode_binary(buf->buff, index, nullptr, &len) == 0) {
		char *data = new char[len];
		ei_decode_binary(buf->buff, index, data, &len);
		String result = String::utf8(data, (int)len);
		delete[] data;
		return result;
	}

	return Variant();
}

Variant ErlangVariantDecoder::decode_tuple(ei_x_buff *buf, int *index, int arity, BeamObjectScope *scope) {
	if (arity < 1) {
		return Variant();
	}

	char type_atom[256];
	if (ei_decode_atom(buf->buff, index, type_atom) < 0) {
		return Variant();
	}

	if (strcmp(type_atom, "vector2") == 0 && arity == 3) {
		return decode_vector2(buf, index);
	} else if (strcmp(type_atom, "vector2i") == 0 && arity == 3) {
		return decode_vector2i(buf, index);
	} else if (strcmp(type_atom, "rect2") == 0 && arity == 5) {
		return decode_rect2(buf, index);
	} else if (strcmp(type_atom, "rect2i") == 0 && arity == 5) {
		return decode_rect2i(buf, index);
	} else if (strcmp(type_atom, "vector3") == 0 && arity == 4) {
		return decode_vector3(buf, index);
	} else if (strcmp(type_atom, "vector3i") == 0 && arity == 4) {
		return decode_vector3i(buf, index);
	} else if (strcmp(type_atom, "transform2d") == 0 && arity == 7) {
		return decode_transform2d(buf, index);
	} else if (strcmp(type_atom, "vector4") == 0 && arity == 5) {
		return decode_vector4(buf, index);
	} else if (strcmp(type_atom, "vector4i") == 0 && arity == 5) {
		return decode_vector4i(buf, index);
	} else if (strcmp(type_atom, "plane") == 0 && arity == 5) {
		return decode_plane(buf, index);
	} else if (strcmp(type_atom, "quaternion") == 0 && arity == 5) {
		return decode_quaternion(buf, index);
	} else if (strcmp(type_atom, "aabb") == 0 && arity == 7) {
		return decode_aabb(buf, index);
	} else if (strcmp(type_atom, "basis") == 0 && arity == 10) {
		return decode_basis(buf, index);
	} else if (strcmp(type_atom, "transform3d") == 0 && arity == 13) {
		return decode_transform3d(buf, index);
	} else if (strcmp(type_atom, "projection") == 0 && arity == 17) {
		return decode_projection(buf, index);
	} else if (strcmp(type_atom, "color") == 0 && arity == 5) {
		return decode_color(buf, index);
	} else if (strcmp(type_atom, "string_name") == 0 && arity == 2) {
		return decode_string_name(buf, index);
	} else if (strcmp(type_atom, "node_path") == 0 && arity == 2) {
		return decode_node_path(buf, index);
	} else if (strcmp(type_atom, "rid") == 0 && arity == 2) {
		return decode_rid(buf, index);
	} else if (strcmp(type_atom, "callable") == 0 && arity == 2) {
		return decode_callable(buf, index);
	} else if (strcmp(type_atom, "signal") == 0 && arity == 2) {
		return decode_signal(buf, index);
	} else if (strcmp(type_atom, "dictionary") == 0 && arity == 2) {
		return decode_dictionary(buf, index, scope);
	} else if (strcmp(type_atom, "array") == 0 && arity == 2) {
		return decode_array(buf, index, scope);
	} else if (strcmp(type_atom, "object") == 0 && arity == 3) {
		return decode_object(buf, index, scope);
	}

	return Variant();
}

Variant ErlangVariantDecoder::decode_vector2(ei_x_buff *buf, int *index) {
	double x, y;
	if (ei_decode_double(buf->buff, index, &x) < 0 ||
			ei_decode_double(buf->buff, index, &y) < 0) {
		return Variant();
	}
	return Variant(Vector2(x, y));
}

Variant ErlangVariantDecoder::decode_vector3(ei_x_buff *buf, int *index) {
	double x, y, z;
	if (ei_decode_double(buf->buff, index, &x) < 0 ||
			ei_decode_double(buf->buff, index, &y) < 0 ||
			ei_decode_double(buf->buff, index, &z) < 0) {
		return Variant();
	}
	return Variant(Vector3(x, y, z));
}

Variant ErlangVariantDecoder::decode_vector4(ei_x_buff *buf, int *index) {
	double x, y, z, w;
	if (ei_decode_double(buf->buff, index, &x) < 0 ||
			ei_decode_double(buf->buff, index, &y) < 0 ||
			ei_decode_double(buf->buff, index, &z) < 0 ||
			ei_decode_double(buf->buff, index, &w) < 0) {
		return Variant();
	}
	return Variant(Vector4(x, y, z, w));
}

Variant ErlangVariantDecoder::decode_color(ei_x_buff *buf, int *index) {
	double r, g, b, a;
	if (ei_decode_double(buf->buff, index, &r) < 0 ||
			ei_decode_double(buf->buff, index, &g) < 0 ||
			ei_decode_double(buf->buff, index, &b) < 0 ||
			ei_decode_double(buf->buff, index, &a) < 0) {
		return Variant();
	}
	return Variant(Color(r, g, b, a));
}

Variant ErlangVariantDecoder::decode_vector2i(ei_x_buff *buf, int *index) {
	long x, y;
	if (ei_decode_long(buf->buff, index, &x) < 0 ||
			ei_decode_long(buf->buff, index, &y) < 0) {
		return Variant();
	}
	return Variant(Vector2i(x, y));
}

Variant ErlangVariantDecoder::decode_rect2(ei_x_buff *buf, int *index) {
	double px, py, sx, sy;
	if (ei_decode_double(buf->buff, index, &px) < 0 ||
			ei_decode_double(buf->buff, index, &py) < 0 ||
			ei_decode_double(buf->buff, index, &sx) < 0 ||
			ei_decode_double(buf->buff, index, &sy) < 0) {
		return Variant();
	}
	return Variant(Rect2(px, py, sx, sy));
}

Variant ErlangVariantDecoder::decode_rect2i(ei_x_buff *buf, int *index) {
	long px, py, sx, sy;
	if (ei_decode_long(buf->buff, index, &px) < 0 ||
			ei_decode_long(buf->buff, index, &py) < 0 ||
			ei_decode_long(buf->buff, index, &sx) < 0 ||
			ei_decode_long(buf->buff, index, &sy) < 0) {
		return Variant();
	}
	return Variant(Rect2i(px, py, sx, sy));
}

Variant ErlangVariantDecoder::decode_vector3i(ei_x_buff *buf, int *index) {
	long x, y, z;
	if (ei_decode_long(buf->buff, index, &x) < 0 ||
			ei_decode_long(buf->buff, index, &y) < 0 ||
			ei_decode_long(buf->buff, index, &z) < 0) {
		return Variant();
	}
	return Variant(Vector3i(x, y, z));
}

Variant ErlangVariantDecoder::decode_transform2d(ei_x_buff *buf, int *index) {
	double m11, m12, m21, m22, ox, oy;
	if (ei_decode_double(buf->buff, index, &m11) < 0 ||
			ei_decode_double(buf->buff, index, &m12) < 0 ||
			ei_decode_double(buf->buff, index, &m21) < 0 ||
			ei_decode_double(buf->buff, index, &m22) < 0 ||
			ei_decode_double(buf->buff, index, &ox) < 0 ||
			ei_decode_double(buf->buff, index, &oy) < 0) {
		return Variant();
	}
	return Variant(Transform2D(Vector2(m11, m12), Vector2(m21, m22), Vector2(ox, oy)));
}

Variant ErlangVariantDecoder::decode_vector4i(ei_x_buff *buf, int *index) {
	long x, y, z, w;
	if (ei_decode_long(buf->buff, index, &x) < 0 ||
			ei_decode_long(buf->buff, index, &y) < 0 ||
			ei_decode_long(buf->buff, index, &z) < 0 ||
			ei_decode_long(buf->buff, index, &w) < 0) {
		return Variant();
	}
	return Variant(Vector4i(x, y, z, w));
}

Variant ErlangVariantDecoder::decode_plane(ei_x_buff *buf, int *index) {
	double nx, ny, nz, d;
	if (ei_decode_double(buf->buff, index, &nx) < 0 ||
			ei_decode_double(buf->buff, index, &ny) < 0 ||
			ei_decode_double(buf->buff, index, &nz) < 0 ||
			ei_decode_double(buf->buff, index, &d) < 0) {
		return Variant();
	}
	return Variant(Plane(nx, ny, nz, d));
}

Variant ErlangVariantDecoder::decode_quaternion(ei_x_buff *buf, int *index) {
	double x, y, z, w;
	if (ei_decode_double(buf->buff, index, &x) < 0 ||
			ei_decode_double(buf->buff, index, &y) < 0 ||
			ei_decode_double(buf->buff, index, &z) < 0 ||
			ei_decode_double(buf->buff, index, &w) < 0) {
		return Variant();
	}
	return Variant(Quaternion(x, y, z, w));
}

Variant ErlangVariantDecoder::decode_aabb(ei_x_buff *buf, int *index) {
	double px, py, pz, sx, sy, sz;
	if (ei_decode_double(buf->buff, index, &px) < 0 ||
			ei_decode_double(buf->buff, index, &py) < 0 ||
			ei_decode_double(buf->buff, index, &pz) < 0 ||
			ei_decode_double(buf->buff, index, &sx) < 0 ||
			ei_decode_double(buf->buff, index, &sy) < 0 ||
			ei_decode_double(buf->buff, index, &sz) < 0) {
		return Variant();
	}
	return Variant(AABB(Vector3(px, py, pz), Vector3(sx, sy, sz)));
}

Variant ErlangVariantDecoder::decode_basis(ei_x_buff *buf, int *index) {
	double r0x, r0y, r0z, r1x, r1y, r1z, r2x, r2y, r2z;
	if (ei_decode_double(buf->buff, index, &r0x) < 0 ||
			ei_decode_double(buf->buff, index, &r0y) < 0 ||
			ei_decode_double(buf->buff, index, &r0z) < 0 ||
			ei_decode_double(buf->buff, index, &r1x) < 0 ||
			ei_decode_double(buf->buff, index, &r1y) < 0 ||
			ei_decode_double(buf->buff, index, &r1z) < 0 ||
			ei_decode_double(buf->buff, index, &r2x) < 0 ||
			ei_decode_double(buf->buff, index, &r2y) < 0 ||
			ei_decode_double(buf->buff, index, &r2z) < 0) {
		return Variant();
	}
	return Variant(Basis(Vector3(r0x, r0y, r0z), Vector3(r1x, r1y, r1z), Vector3(r2x, r2y, r2z)));
}

Variant ErlangVariantDecoder::decode_transform3d(ei_x_buff *buf, int *index) {
	double r0x, r0y, r0z, r1x, r1y, r1z, r2x, r2y, r2z, ox, oy, oz;
	if (ei_decode_double(buf->buff, index, &r0x) < 0 ||
			ei_decode_double(buf->buff, index, &r0y) < 0 ||
			ei_decode_double(buf->buff, index, &r0z) < 0 ||
			ei_decode_double(buf->buff, index, &r1x) < 0 ||
			ei_decode_double(buf->buff, index, &r1y) < 0 ||
			ei_decode_double(buf->buff, index, &r1z) < 0 ||
			ei_decode_double(buf->buff, index, &r2x) < 0 ||
			ei_decode_double(buf->buff, index, &r2y) < 0 ||
			ei_decode_double(buf->buff, index, &r2z) < 0 ||
			ei_decode_double(buf->buff, index, &ox) < 0 ||
			ei_decode_double(buf->buff, index, &oy) < 0 ||
			ei_decode_double(buf->buff, index, &oz) < 0) {
		return Variant();
	}
	Basis basis(Vector3(r0x, r0y, r0z), Vector3(r1x, r1y, r1z), Vector3(r2x, r2y, r2z));
	return Variant(Transform3D(basis, Vector3(ox, oy, oz)));
}

Variant ErlangVariantDecoder::decode_projection(ei_x_buff *buf, int *index) {
	Projection p;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			double val;
			if (ei_decode_double(buf->buff, index, &val) < 0) {
				return Variant();
			}
			p.columns[i][j] = val;
		}
	}
	return Variant(p);
}

Variant ErlangVariantDecoder::decode_string_name(ei_x_buff *buf, int *index) {
	long len = 0;
	if (ei_decode_binary(buf->buff, index, nullptr, &len) < 0) {
		return Variant();
	}
	char *data = new char[len];
	ei_decode_binary(buf->buff, index, data, &len);
	String str = String::utf8(data, (int)len);
	delete[] data;
	return Variant(StringName(str));
}

Variant ErlangVariantDecoder::decode_node_path(ei_x_buff *buf, int *index) {
	long len = 0;
	if (ei_decode_binary(buf->buff, index, nullptr, &len) < 0) {
		return Variant();
	}
	char *data = new char[len];
	ei_decode_binary(buf->buff, index, data, &len);
	String str = String::utf8(data, (int)len);
	delete[] data;
	return Variant(NodePath(str));
}

Variant ErlangVariantDecoder::decode_rid(ei_x_buff *buf, int *index) {
	long id;
	if (ei_decode_long(buf->buff, index, &id) < 0) {
		return Variant();
	}
	return Variant(RID::from_uint64(id));
}

Variant ErlangVariantDecoder::decode_callable(ei_x_buff *buf, int *index) {
	// For now, skip the unsupported marker and return empty callable
	char atom[256];
	ei_decode_atom(buf->buff, index, atom);
	return Variant(Callable());
}

Variant ErlangVariantDecoder::decode_signal(ei_x_buff *buf, int *index) {
	// For now, skip the unsupported marker and return empty signal
	char atom[256];
	ei_decode_atom(buf->buff, index, atom);
	return Variant(Signal());
}

Variant ErlangVariantDecoder::decode_dictionary(ei_x_buff *buf, int *index, BeamObjectScope *scope) {
	int arity;
	if (ei_decode_list_header(buf->buff, index, &arity) < 0) {
		return Variant();
	}

	Dictionary dict;
	for (int i = 0; i < arity; i++) {
		int tuple_arity;
		if (ei_decode_tuple_header(buf->buff, index, &tuple_arity) < 0 || tuple_arity != 2) {
			return Variant();
		}
		Variant key = decode_variant(buf, index, scope);
		Variant value = decode_variant(buf, index, scope);
		dict[key] = value;
	}

	return Variant(dict);
}

Variant ErlangVariantDecoder::decode_array(ei_x_buff *buf, int *index, BeamObjectScope *scope) {
	int arity;
	if (ei_decode_list_header(buf->buff, index, &arity) < 0) {
		return Variant();
	}

	Array arr;
	for (int i = 0; i < arity; i++) {
		Variant value = decode_variant(buf, index, scope);
		arr.push_back(value);
	}

	// Skip empty list terminator
	ei_skip_term(buf->buff, index);
	return Variant(arr);
}

Variant ErlangVariantDecoder::decode_packed_array(ei_x_buff *buf, int *index, const char *type_atom) {
	// For now, skip the empty list and return empty packed array
	int arity;
	ei_decode_list_header(buf->buff, index, &arity);
	ei_skip_term(buf->buff, index);

	// Return appropriate empty packed array based on type
	if (strcmp(type_atom, "packed_byte_array") == 0) {
		return Variant(PackedByteArray());
	} else if (strcmp(type_atom, "packed_int32_array") == 0) {
		return Variant(PackedInt32Array());
	} else if (strcmp(type_atom, "packed_int64_array") == 0) {
		return Variant(PackedInt64Array());
	} else if (strcmp(type_atom, "packed_float32_array") == 0) {
		return Variant(PackedFloat32Array());
	} else if (strcmp(type_atom, "packed_float64_array") == 0) {
		return Variant(PackedFloat64Array());
	} else if (strcmp(type_atom, "packed_string_array") == 0) {
		return Variant(PackedStringArray());
	} else if (strcmp(type_atom, "packed_vector2_array") == 0) {
		return Variant(PackedVector2Array());
	} else if (strcmp(type_atom, "packed_vector3_array") == 0) {
		return Variant(PackedVector3Array());
	} else if (strcmp(type_atom, "packed_color_array") == 0) {
		return Variant(PackedColorArray());
	} else if (strcmp(type_atom, "packed_vector4_array") == 0) {
		return Variant(PackedVector4Array());
	}
	return Variant();
}

Variant ErlangVariantDecoder::decode_object(ei_x_buff *buf, int *index, BeamObjectScope *scope) {
	long obj_ptr;
	long len = 0;

	if (ei_decode_long(buf->buff, index, &obj_ptr) < 0) {
		return Variant();
	}

	if (ei_decode_binary(buf->buff, index, nullptr, &len) < 0) {
		return Variant();
	}

	char *class_name_data = new char[len];
	ei_decode_binary(buf->buff, index, class_name_data, &len);
	delete[] class_name_data;

	// Validate object pointer if scope is provided
	if (scope && !scope->is_object_scoped((uintptr_t)obj_ptr)) {
		return Variant();
	}

	Object *obj = reinterpret_cast<Object *>(obj_ptr);
	return Variant(obj);
}

#endif // LINUXBSD_ENABLED
