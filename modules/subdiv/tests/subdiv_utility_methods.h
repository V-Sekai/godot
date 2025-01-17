/**************************************************************************/
/*  subdiv_utility_methods.h                                              */
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

#ifndef SUBDIV_UTILITY_METHODS_H
#define SUBDIV_UTILITY_METHODS_H

#include "core/variant/typed_array.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool contains_default(PackedVector3Array arr) {
	for (int i = 0; i < arr.size(); i++) {
		if (arr[i] == Vector3(0, 0, 0)) {
			return true;
		}
	}
	return false;
}

bool equal_approx(PackedVector3Array result, PackedVector3Array expected) {
	if (result.size() != expected.size()) {
		return false;
	}

	for (int i = 0; i < result.size(); i++) {
		if (!result[i].is_equal_approx(expected[i])) {
			return false;
		}
	}
	return true;
}

PackedInt32Array create_packed_int32_array(int32_t arr[], int size) {
	PackedInt32Array packed_array;
	for (int i = 0; i < size; i++) {
		packed_array.push_back(arr[i]);
	}
	return packed_array;
}

#endif // SUBDIV_UTILITY_METHODS_H
