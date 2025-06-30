/**************************************************************************/
/*  object_weak_ref.h                                                     */
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

#include "classes/object.h"

namespace zylann::godot {

// Holds a weak reference to a Godot object.
// Mainly useful to reference scene tree nodes more safely, because their ownership model is harder to handle with
// pointers, compared to RefCounted objects.
// It is not intended for use with RefCounted objects.
// Warning: if the object can be destroyed by a different thread, this won't be safe to use.
template <typename T>
class ObjectWeakRef {
public:
	void set(T *obj) {
		_id = obj != nullptr ? obj->get_instance_id() : ObjectID();
	}

	T *get() const {
		if (!_id.is_valid()) {
			return nullptr;
		}
		Object *obj = ObjectDB::get_instance(_id);
		if (obj == nullptr) {
			// Could have been destroyed.
			// _node_object_id = ObjectID();
			return nullptr;
		}
		T *tobj = Object::cast_to<T>(obj);
		// We don't expect Godot to reuse the same ObjectID for different objects
		ERR_FAIL_COND_V(tobj == nullptr, nullptr);
		return tobj;
	}

private:
	ObjectID _id;
};

} // namespace zylann::godot
