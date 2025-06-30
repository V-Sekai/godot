/**************************************************************************/
/*  spatial_lock_2d.cpp                                                   */
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

#include "spatial_lock_2d.h"
#include "../io/log.h"
#include "../string/format.h"

namespace zylann::voxel {

SpatialLock2D::SpatialLock2D() {
	_boxes.reserve(8);
}

void SpatialLock2D::remove_box(const BoxBounds2i &box, Mode mode) {
#ifdef ZN_SPATIAL_LOCK_2D_CHECKS
	const Thread::ID thread_id = Thread::get_caller_id();
#endif

	for (unsigned int i = 0; i < _boxes.size(); ++i) {
		const Box &existing_box = _boxes[i];

		if (existing_box.bounds == box && existing_box.mode == mode
#ifdef ZN_SPATIAL_LOCK_2D_CHECKS
			&& existing_box.thread_id == thread_id
#endif
		) {
			_boxes[i] = _boxes[_boxes.size() - 1];
			_boxes.pop_back();
			return;
		}
	}
	// Could be a bug
	ZN_PRINT_ERROR(format("Could not find box to remove {} with mode {}", box, mode));
}

} // namespace zylann::voxel
