/**************************************************************************/
/*  object.h                                                              */
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

/**
 * object.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
#ifndef COMPAT_OBJECT_H
#define COMPAT_OBJECT_H

#ifdef LIMBOAI_MODULE

#include "core/object/object.h"

#define OBJECT_DB_GET_INSTANCE(m_id) ObjectDB::get_instance(ObjectID(m_id))
#define IS_CLASS(m_obj, m_class) (m_obj->is_class_ptr(m_class::get_class_ptr_static()))
#define GET_SCRIPT(m_obj) (m_obj->get_script_instance() ? m_obj->get_script_instance()->get_script() : nullptr)

_FORCE_INLINE_ bool OBJECT_HAS_PROPERTY(Object *p_obj, const StringName &p_prop) {
	bool r_valid;
	return Variant(p_obj).has_key(p_prop, r_valid);
}

#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION

#include "godot_cpp/core/object.hpp"

#define OBJECT_DB_GET_INSTANCE(m_id) ObjectDB::get_instance(m_id)
// TODO: Use this def if https://github.com/godotengine/godot-cpp/pull/1356 gets merged:
// #define IS_CLASS(m_obj, m_class) (m_obj->is_class_static(m_class::get_class_static()))
#define IS_CLASS(m_obj, m_class) (m_obj->is_class(#m_class))
#define GET_SCRIPT(m_obj) (m_obj->get_script())

_FORCE_INLINE_ bool OBJECT_HAS_PROPERTY(Object *p_obj, const StringName &p_prop) {
	return Variant(p_obj).has_key(p_prop);
}

#endif // LIMBOAI_GDEXTENSION

#endif // COMPAT_OBJECT_H
